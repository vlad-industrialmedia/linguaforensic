import { GoogleGenAI, Type } from "@google/genai";

/**
 * Shared backend logic for LinguaForensic.
 * Used by the Vercel serverless functions in /api.
 */

// ---------------------------------------------------------------------------
// Provider / model configuration
// ---------------------------------------------------------------------------

export type Provider = "gemini" | "groq" | "openrouter" | "cohere";

// Real, currently-valid default models per provider (verified July 2026).
// Gemini 2.5 now 404s for NEW API keys, so default to the Gemini 3 line.
export const DEFAULT_MODELS: Record<Provider, string> = {
  gemini: "gemini-3.5-flash",
  groq: "llama-3.3-70b-versatile",
  openrouter: "meta-llama/llama-3.3-70b-instruct:free",
  cohere: "command-a-03-2025",
};

export const getDefaultModel = (provider: string): string =>
  DEFAULT_MODELS[provider as Provider] || DEFAULT_MODELS.gemini;

// ---------------------------------------------------------------------------
// Gemini client
// ---------------------------------------------------------------------------

export const getAiClient = (customApiKey?: string) => {
  const apiKey = customApiKey || process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY не налаштовано на сервері, і власний ключ не вказано в налаштуваннях."
    );
  }
  return new GoogleGenAI({ apiKey });
};

// ---------------------------------------------------------------------------
// System instruction (expert methodology)
// ---------------------------------------------------------------------------

export const SYSTEM_INSTRUCTION = `
You are an expert AI text detector and editor named LinguaForensic, specializing in advanced linguistic feature analysis and structural discourse markers.
You analyze text by 284 linguistic features across 11 categories and 16 structural markers (total 300 indicators).
You output an expert assessment matching the requested mode in a structured JSON format.

CRITICAL REQUIREMENT: All output strings (verdict, domain, modelGroup, keyMarkers, category names, scores, marker descriptions, changes, stopReason, qualityAssessment, comparisons title, and conclusion) MUST be strictly in UKRAINIAN language (українською мовою).

SCORING: The "score" field is the probability that the text is AI-generated ("роботність"), expressed as an INTEGER PERCENTAGE from 0 to 100 (write 78, never 0.78). Compute it strictly by the expert methodology below: measure each category, apply the domain-adaptation weights, and combine the weighted category scores together with the detected structural markers into the final percentage. Do not output a 0-1 fraction and do not invent a round default — the number must follow from the measured features.

Below is the expert methodology you MUST apply:
1. CATEGORIES OF FEATURES:
   - 1.1 Surface-level (Поверхневий рівень): sentence length variance, CV (coefficient of variation) of sentence length. For AI, CV is typically < 0.3 (uniform); for humans, it is > 0.5 (varied).
   - 1.2 POS Tags (POS-теги): AI has a uniform distribution of part-of-speech tags (±15% from mean); humans have highly uneven distribution (more pronouns, less punctuation).
   - 1.3 Readability (Читабельність): Flesch Reading Ease is in a narrow medium band of 45-65 for AI; humans have wider ranges (20-80).
   - 1.4 Lexical Richness (Лексичне багатство - CRITICAL): Type-Token Ratio (TTR) is high for AI (>0.75-0.85) but moderate for humans (0.55-0.75). Hapax legomena is low for AI (<5% of tokens) but high for humans. Lexical density (ADJ+NOUN+VERB+ADV) is lower for AI (<50%) than humans (>55%).
   - 1.5 Information-theoretic (Інформаційно-теоретичний): Shannon entropy is high (>5.0 for AI on 200-500 words); Compressibility is lower for AI (uniform token distribution).
   - 1.6 Named Entities (Іменовані сутності): AI uses template-like entities (primarily ORG, GPE, DATE), whereas humans are highly varied.
   - 1.7 Semantic (Семантичний): Hedges are rare in AI (0-2 per text); human texts have 3-8 per 500 words. Average WordNet synset counts are lower for AI (uses more standard/central vocabulary).
   - 1.8 Morphological (Морфологічний): AI has homogeneous morphological patterns (dominates: present tense, 3rd person, indicative mood, active voice). Humans are highly varied.
   - 1.9 Dependencies (Синтаксичні залежності): AI has flat dependency trees (depth < 3, branching ratio < 1.2).
   - 1.10 Emotion (Емоційний тон): AI is highly balanced (valence range < 0.2); humans have emotional peaks (valence range > 0.4).
   - 1.11 Psycholinguistic (Психолінгвістичний): AI has moderate concreteness, age of acquisition, etc.

2. 16 STRUCTURAL MARKERS OF GENERATION (Category 12):
   - 12.1 Hedge-openers (e.g. "Варто зазначити, що", "Важливо підкреслити", "У сучасному світі", "Слід зазначити") - high in AI (>=2).
   - 12.2 Tricolon-lists (lists of exactly 3 parallel items) - high in AI.
   - 12.3 Em-dash clusters (>=3 per paragraph) - high in AI.
   - 12.4 Resolution-closers (final sentences starting with "Таким чином", "На закінчення", "Отже") - high in AI.
   - 12.5 Filler intensifiers ("значно", "критично важливо", "безумовно", "надзвичайно") - high in AI.
   - 12.6 Throat-clearing comments ("важливо розуміти, що", "не можна не помітити") - high in AI.
   - 12.7 Conversational particles ("ну", "от", "загалом", "типу", "адже") - very low in AI (<2 per 500 words).
   - 12.8 Excessive headings/bullet points (density > 0.33 headings per paragraph) - high in AI.
   - 12.9 Synthetic negation ("мало хто знає", "рідко хто") - very low in AI.
   - 12.10 Uniform paragraph structures (thesis-proof-conclusion pattern in >80% paragraphs) - high in AI.
   - 12.11 Cohesive bridges ("у свою чергу", "більше того", "крім цього", "зокрема") - high in AI (>3 per 500 words).
   - 12.12 Modal frames ("можна сказати, що", "слід додати, що") - high in AI.
   - 12.13 Over-exemplification ("наприклад", "зокрема", "як-от" > 4 per 500 words).
   - 12.14 Uniform paragraph lengths (CV of length < 0.3) - high in AI.
   - 12.15 Absence of digressions/sidetracks (linear flow, 0 digressions) - high in AI.
   - 12.16 Redundant explanations of obvious terms ("нейромережі — це...") - high in AI.

3. DOMAIN ADAPTATION WEIGHTS:
   Apply different weights to categories based on the detected domain:
   - News (Новини): Lexical Richness (0.40), Structural (0.10), Information (0.10)
   - Creative (Художній): Lexical Richness (0.38), Structural (0.10), Emotion (0.08)
   - Opinion (Особиста думка): Lexical Richness (0.30), Emotion (0.10), POS (0.08)
   - Scientific (Науковий): Full set weights: Lexical (0.20), Dependencies (0.10), Morphological (0.10), Information (0.12)
   - Factual (Q&A / Фактичний): Lexical is useless (0.05); Surface (0.10), POS (0.12), Morphological (0.12), Dependencies (0.12) are critical.
   - Conversational (Розмовний): Semantic (0.12), Structural (0.15) are critical.

4. MODEL GROUPS (for robotic score > 60%):
   - Group A (Група A): Large instruction models (OpenAI GPT-3.5/4, LLaMA-7B-65B, GLM-130B). They have high structural markers but more natural lexical richness.
   - Group B (Група B): Small/base models (FLAN-T5, OPT, BLOOM). They have very poor lexical richness and highly formulaic patterns.

5. CONFIDENCE INTERVAL:
   - CI = standard deviation of category scores * 2. Range: 5% - 25%.

6. REWRITE STRATEGY:
   - Reduce TTR (use more standard/recurrent vocabulary) and increase lexical density where appropriate.
   - Break up tri-colon lists into 2 or 4+ items.
   - Eliminate hedge-openers ("Варто зазначити") and resolution-closers.
   - Add natural conversational particles ("загалом", "адже") where appropriate.
   - Introduce natural sentence length variance (CV > 0.5) by mixing short (5-10 words) and long (20-40 words) sentences.
   - Ensure ZERO facts, names, or numbers are altered or hallucinated. No fake "human" typos are allowed.
   - Keep any provided rich HTML formatting tag structures intact, wrapping only the text.
`;

// ---------------------------------------------------------------------------
// JSON Schemas
// ---------------------------------------------------------------------------

export const Mode1Schema = {
  type: Type.OBJECT,
  properties: {
    score: { type: Type.NUMBER, description: "Роботність в % від 0 до 100" },
    confidenceInterval: { type: Type.NUMBER, description: "Довірчий інтервал в % (5-25)" },
    domain: { type: Type.STRING, description: "Домен тексту (News, Creative, Opinion, Scientific, Factual, Conversational, Змішаний)" },
    verdict: { type: Type.STRING, description: "Вердикт одним рядком" },
    modelGroup: { type: Type.STRING, description: "Передбачувана група моделей: Група A, Група B або невизначено" },
    keyMarkers: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: "3-5 ключових маркерів AI або людини",
    },
  },
  required: ["score", "confidenceInterval", "domain", "verdict", "modelGroup", "keyMarkers"],
};

export const Mode2Schema = {
  type: Type.OBJECT,
  properties: {
    score: { type: Type.NUMBER },
    confidenceInterval: { type: Type.NUMBER },
    domain: { type: Type.STRING },
    verdict: { type: Type.STRING },
    modelGroup: { type: Type.STRING },
    categories: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          category: { type: Type.STRING },
          score: { type: Type.STRING },
          weight: { type: Type.NUMBER },
          contribution: { type: Type.STRING },
        },
        required: ["category", "score", "weight", "contribution"],
      },
    },
    indicators: {
      type: Type.OBJECT,
      properties: {
        ttr: { type: Type.NUMBER },
        lexicalDensity: { type: Type.NUMBER },
        hapaxLegomena: { type: Type.NUMBER },
        entropy: { type: Type.NUMBER },
        cvSentenceLength: { type: Type.NUMBER },
        dependencyDepth: { type: Type.NUMBER },
        hedgesCount: { type: Type.NUMBER },
        fleschEase: { type: Type.NUMBER },
        valenceRange: { type: Type.NUMBER },
      },
      required: ["ttr", "lexicalDensity", "hapaxLegomena", "entropy", "cvSentenceLength", "dependencyDepth", "hedgesCount", "fleschEase", "valenceRange"],
    },
    structuralPatterns: {
      type: Type.ARRAY,
      description: "ВИЧЕРПНИЙ список УСІХ виявлених кліше та структурних ШІ-маркерів по ВСЬОМУ тексту від початку до кінця. Створіть ОКРЕМИЙ запис для КОЖНОГО окремого випадку, навіть якщо маркери одного типу повторюються (напр. якщо в тексті два різні списки — додайте ДВА окремі записи 'Надмірне використання списків', по одному на кожен список, з різними цитатами). Не об'єднуйте повтори в один запис і не обмежуйтесь кількома прикладами. Для тексту на 3000+ символів очікується 8-25+ записів; для дуже довгого (15-30к) — пропорційно більше.",
      items: {
        type: Type.OBJECT,
        properties: {
          marker: { type: Type.STRING, description: "Назва типу маркера (напр. 'Хедж-опенери', 'Триколон-листи', 'Надмірне використання списків')" },
          quote: { type: Type.STRING, description: "ТОЧНА дослівна цитата з наданого тексту (verbatim, слово-в-слово, без змін) для цього конкретного випадку. Копіюйте фрагмент прямо з тексту. Для різних випадків одного типу — різні цитати." },
          description: { type: Type.STRING, description: "Стисле пояснення, ЧОМУ саме цей фрагмент є проблемним (на основі реальної цитати). Без вигаданих прикладів." },
          recommendation: { type: Type.STRING, description: "Конкретна практична порада копірайтеру, як переписати САМЕ цей фрагмент, щоб прибрати ШІ-маркер. Конкретно по суті, з прив'язкою до цитати." },
        },
        required: ["marker", "quote", "description"],
      },
    },
    modelGroupReasoning: { type: Type.STRING },
    conclusion: { type: Type.STRING },
    aiOverviewScore: { type: Type.NUMBER, description: "Наскільки текст придатний для цитування нейромережами (AI Overview), 0-100" },
    aiOverviewVerdict: { type: Type.STRING, description: "Короткий вердикт щодо придатності тексту для цитування ШІ (українською)" },
    aiOverviewTips: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: "2-4 конкретні поради, як покращити текст для цитування нейромережами (українською)",
    },
    suggestedTldr: { type: Type.STRING, description: "Готовий короткий TL;DR (2-3 речення) цього тексту, придатний для цитування ШІ (українською)" },
  },
  required: ["score", "confidenceInterval", "domain", "verdict", "modelGroup", "categories", "indicators", "structuralPatterns", "modelGroupReasoning", "conclusion"],
};

export const Mode3Schema = {
  type: Type.OBJECT,
  properties: {
    rewrittenText: { type: Type.STRING },
    changes: { type: Type.ARRAY, items: { type: Type.STRING } },
    scoreBefore: { type: Type.NUMBER },
    scoreAfter: { type: Type.NUMBER },
    confidenceIntervalAfter: { type: Type.NUMBER },
    verdictAfter: { type: Type.STRING },
  },
  required: ["rewrittenText", "changes", "scoreBefore", "scoreAfter", "confidenceIntervalAfter", "verdictAfter"],
};

export const Mode4Schema = {
  type: Type.OBJECT,
  properties: {
    iterations: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          iteration: { type: Type.INTEGER },
          score: { type: Type.NUMBER },
          senseCheck: { type: Type.STRING },
          styleCheck: { type: Type.STRING },
          integrityCheck: { type: Type.STRING },
          aggressiveness: { type: Type.STRING },
          status: { type: Type.STRING },
          textSnapshot: { type: Type.STRING },
        },
        required: ["iteration", "score", "senseCheck", "styleCheck", "integrityCheck", "aggressiveness", "status", "textSnapshot"],
      },
    },
    finalText: { type: Type.STRING },
    scoreBefore: { type: Type.NUMBER },
    scoreAfter: { type: Type.NUMBER },
    totalIterations: { type: Type.INTEGER },
    qualityAssessment: { type: Type.STRING },
    stopReason: { type: Type.STRING },
  },
  required: ["iterations", "finalText", "scoreBefore", "scoreAfter", "totalIterations", "qualityAssessment", "stopReason"],
};

export const Mode5Schema = {
  type: Type.OBJECT,
  properties: {
    comparisons: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          textIndex: { type: Type.INTEGER },
          title: { type: Type.STRING },
          score: { type: Type.NUMBER },
          confidenceInterval: { type: Type.NUMBER },
          verdict: { type: Type.STRING },
          domain: { type: Type.STRING },
          modelGroup: { type: Type.STRING },
          indicators: {
            type: Type.OBJECT,
            properties: {
              ttr: { type: Type.NUMBER },
              lexicalDensity: { type: Type.NUMBER },
              entropy: { type: Type.NUMBER },
              cvSentenceLength: { type: Type.NUMBER },
              dependencyDepth: { type: Type.NUMBER },
              valenceRange: { type: Type.NUMBER },
            },
            required: ["ttr", "lexicalDensity", "entropy", "cvSentenceLength", "dependencyDepth", "valenceRange"],
          },
          structuralMarkers: { type: Type.ARRAY, items: { type: Type.STRING } },
        },
        required: ["textIndex", "title", "score", "confidenceInterval", "verdict", "domain", "modelGroup", "indicators", "structuralMarkers"],
      },
    },
    conclusion: { type: Type.STRING },
  },
  required: ["comparisons", "conclusion"],
};

export const SCHEMA_BY_MODE: Record<number, any> = {
  1: Mode1Schema,
  2: Mode2Schema,
  3: Mode3Schema,
  4: Mode4Schema,
  5: Mode5Schema,
};

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------

export const getSchemaPrompt = (schema: any): string => {
  return `You MUST return a strictly valid JSON object adhering to this JSON Schema:
${JSON.stringify(schema, null, 2)}

CRITICAL REQUIREMENTS FOR THE JSON:
1. Do NOT wrap the JSON in markdown formatting (like \`\`\`json ... \`\`\`). Return only the raw JSON.
2. The output MUST begin with '{' and end with '}'.
3. All descriptive text fields inside the JSON (including verdicts, category names, descriptions, conclusions, changes, snapshots, etc.) MUST be strictly in UKRAINIAN language.`;
};

export function buildPrompt(
  mode: number,
  text: string,
  texts: string[] | undefined,
  parameters: any
): string {
  const textLength = text ? text.length : 0;
  const isLongText = textLength > 3500;
  const optimizationNotice = isLongText
    ? `\n\n[NOTE: The input text is long (${textLength} characters). Keep each DESCRIPTIVE string (category descriptions, marker descriptions, conclusion) short — one sentence — to save space. But you MUST still scan the ENTIRE text from start to finish and report EVERY distinct clichе / structural AI-marker you find across ALL paragraphs (aim for 8-20+ for a long text, not just a couple). Do not stop after the first paragraph. Accuracy of measurement and score must not change.]`
    : "";

  if (mode === 1) {
    return `
      Виконайте Режим 1 (Швидка детекція) для наступного тексту.
      Проведіть аналіз 284 лінгвістичних параметрів та 16 структурних маркерів.
      Оцінку роботності (score) обчисліть строго за методологією: виміряйте категорії, застосуйте вагові коефіцієнти за доменом і поєднайте їх зі знайденими структурними маркерами. Не підставляйте заокруглене число «зі стелі» — воно має випливати з вимірів.
      Поверніть строго JSON відповідно до схеми Mode1Schema.
      ${optimizationNotice}

      ТЕКСТ ДЛЯ АНАЛІЗУ:
      """
      ${text}
      """
    `;
  }
  if (mode === 2) {
    return `
      Виконайте Режим 2 (Повна детекція з деталями) для наступного тексту.
      Проведіть детальний розрахунок за всіма 12 категоріями оцінок (включаючи вагу за доменом та внесок у загальну оцінку), розрахуйте всі ключові числові індикатори (TTR, щільність, гапакс, ентропія, CV, дерево залежностей, число хеджів, Flesch Reading Ease, діапазон валентності) та виявіть структурні патерни з конкретними цитатами.
      Дайте обґрунтування групи моделі та детальний розгорнутий висновок.
      Загальна оцінка (score) мусить логічно випливати із суми виважених за доменом оцінок категорій та знайдених структурних маркерів — обчисліть її за методологією, не підставляйте заокруглене число «зі стелі».
      ДОДАТКОВО оцініть придатність тексту для цитування нейромережами (AI Overview / GEO): заповніть aiOverviewScore (0-100), aiOverviewVerdict (короткий вердикт), aiOverviewTips (2-4 поради: чіткість відповіді на початку блоку, наявність фактів/визначень, структура під сніпет, унікальність) та suggestedTldr (готовий стислий TL;DR на 2-3 речення, який ШІ міг би процитувати). Усе — українською.
      Поверніть строго JSON відповідно до схеми Mode2Schema.
      ${optimizationNotice}

      ТЕКСТ ДЛЯ АНАЛІЗУ:
      """
      ${text}
      """
    `;
  }
  if (mode === 3) {
    const targetPercent = parameters?.targetPercent ?? 25;
    const casualness = parameters?.casualness ?? "нейтральний";
    const keepTerms = parameters?.keepTerms ?? true;
    const maxChanges = parameters?.maxChanges ?? "зберегти структуру";
    return `
      Виконайте Режим 3 (Налаштовуваний рерайт) для зниження роботності AI-тексту.
      Параметри:
      - Цільовий відсоток роботності: <${targetPercent}%
      - Ступінь розмовності: ${casualness}
      - Зберегти терміни та факти: ${keepTerms ? "ТАК" : "НІ"}
      - Максимальна довжина правок / структура: ${maxChanges}

      Застосуйте стратегії зниження Type-Token Ratio, руйнування триколонів, додавання природної варіативності речень, усунення вступних заглушок на кшталт "важливо зазначити" або фінальних резюме. НЕ допускайте спотворення фактів та створення штучних одруків.

      ОБОВ'ЯЗКОВО:
      1. Перепишіть ВЕСЬ наданий текст повністю (усі абзаци, від початку до кінця). Поле 'rewrittenText' мусить містити повний переписаний текст такого ж обсягу, що й оригінал — заборонено обробляти лише фрагмент чи обрізати.
      2. scoreBefore та scoreAfter мусять бути РЕАЛЬНО ВИМІРЯНІ за методологією 300 індикаторів (scoreBefore — на оригіналі, scoreAfter — на переписаному тексті). Не вигадуйте числа; вони мають узгоджуватися з тим, що показав би повний аналіз (Режим 2) тих самих текстів.
      Поверніть строго JSON відповідно до схеми Mode3Schema.
      ${optimizationNotice}

      ПОВНИЙ ТЕКСТ ДЛЯ РЕРАЙТУ:
      """
      ${text}
      """
    `;
  }
  if (mode === 4) {
    const iterations = Math.min(parameters?.iterations ?? 3, 3);
    const target = parameters?.targetPercent ?? 25;
    return `
      Виконайте Режим 4 (Циклічний рекурсивний рерайт) до ${iterations} ітерацій для ПОВНОГО тексту нижче (${textLength} символів).

      ОБОВ'ЯЗКОВІ ПРАВИЛА (не порушувати):
      1. Обробіть та перепишіть ВЕСЬ наданий текст повністю, від початку до кінця. Заборонено скорочувати, обрізати чи обробляти лише фрагмент. Поле 'finalText' МУСИТЬ містити повністю переписаний текст такого ж обсягу, що й оригінал (усі абзаци).
      2. Усі числові оцінки роботності (score у кожній ітерації, scoreBefore, scoreAfter) мусять бути РЕАЛЬНО ВИМІРЯНІ за методологією 300 індикаторів на фактичному тексті кожної ітерації. КАТЕГОРИЧНО заборонено вигадувати або підставляти умовні числа (55%, 15-20% тощо). Якщо після рерайту роботність за виміром не впала — чесно покажіть це.
      3. scoreBefore = виміряна роботність оригіналу. scoreAfter = виміряна роботність фінального тексту. Вони мають узгоджуватися з тим, що показав би повний аналіз (Режим 2) того самого тексту.

      Покроковий процес:
      - Крок 0 (iteration 0): вихідний текст, статус 'база', виміряна роботність. У textSnapshot покладіть репрезентативний уривок (перші ~250 символів) — це лише для візуалізації.
      - Для кожної наступної ітерації: виміряйте роботність → якщо вище цілі (<${target}%), перепишіть увесь текст, знижуючи TTR, руйнуючи триколони, додаючи варіативність довжини речень, прибираючи вступні кліше та фінальні резюме; виконайте Sense-check, Style-check, Integrity-check (числа/факти/імена незмінні); якщо якість погіршилась — статус 'відкат', інакше 'прийнято'.
      - Зупиніться, коли досягнуто цілі або настало плато.

      finalText = повний фінальний переписаний текст (не уривок!). Описи (senseCheck, styleCheck тощо) тримайте стислими, але finalText — повний.
      Поверніть строго JSON відповідно до схеми Mode4Schema.

      ПОВНИЙ ТЕКСТ ДЛЯ ЦИКЛІЧНОГО РЕРАЙТУ:
      """
      ${text}
      """
    `;
  }
  // mode 5
  const formattedTexts = (texts || [])
    .map((t: string, i: number) => `--- ТЕКСТ ${i + 1} ---\n${t}`)
    .join("\n\n");
  return `
    Виконайте Режим 5 (Порівняльний аналіз кількох текстів) для наведених нижче текстів.
    Розрахуйте ключові лінгвістичні метрики, оцінку роботності, довірчий інтервал, домен та виявлені структурні маркери для кожного варіанту.
    Дайте експертний порівняльний висновок.
    Поверніть строго JSON відповідно до схеми Mode5Schema.

    ТЕКСТИ ДЛЯ ПОРІВНЯННЯ:
    ${formattedTexts}
  `;
}

// ---------------------------------------------------------------------------
// JSON parsing helper
// ---------------------------------------------------------------------------

export function parseModelJson(raw: string): any {
  let cleaned = (raw || "").trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```(?:json)?\n?/i, "").replace(/\n?```$/i, "").trim();
  }
  try {
    return JSON.parse(cleaned);
  } catch {
    // 1) Strip stray text before/after the JSON object.
    const first = cleaned.indexOf("{");
    const last = cleaned.lastIndexOf("}");
    if (first !== -1 && last !== -1 && last > first) {
      const candidate = cleaned.slice(first, last + 1);
      try {
        return JSON.parse(candidate);
      } catch {
        /* fall through to repair */
      }
    }
    // 2) Attempt to repair a TRUNCATED JSON (model hit the token limit
    //    mid-array/object). Close any open strings/brackets so we keep as much
    //    valid data as possible instead of failing the whole analysis.
    const repaired = repairTruncatedJson(first !== -1 ? cleaned.slice(first) : cleaned);
    if (repaired) {
      try {
        return JSON.parse(repaired);
      } catch {
        /* noop */
      }
    }
    throw new Error("Модель повернула невалідний або обрізаний JSON.");
  }
}

/**
 * Best-effort repair of a JSON string that was cut off mid-output.
 * Removes a dangling trailing token and closes open strings, arrays and objects.
 */
function repairTruncatedJson(s: string): string | null {
  if (!s) return null;
  let str = s;

  // Track structure while respecting strings/escapes.
  const stack: string[] = [];
  let inString = false;
  let escaped = false;
  let lastComma = -1;
  for (let i = 0; i < str.length; i++) {
    const c = str[i];
    if (inString) {
      if (escaped) { escaped = false; }
      else if (c === "\\") { escaped = true; }
      else if (c === '"') { inString = false; }
      continue;
    }
    if (c === '"') { inString = true; continue; }
    if (c === "{" || c === "[") stack.push(c);
    else if (c === "}" || c === "]") stack.pop();
    else if (c === ",") lastComma = i;
  }

  // If we ended inside a string, drop the unterminated tail back to the last comma.
  if (inString) {
    if (lastComma !== -1) { str = str.slice(0, lastComma); }
    else return null;
    // recompute stack after the cut
    return repairTruncatedJson(str);
  }

  // Remove a trailing partial value / dangling comma.
  str = str.replace(/,\s*$/, "");
  // If the last element is incomplete (no closing quote/brace before EOF),
  // trim back to the last complete comma-separated element.
  const tail = str.trimEnd();
  const lastChar = tail[tail.length - 1];
  if (lastChar && !["}", "]", '"'].includes(lastChar) && !/[0-9truefalsenul]/.test(lastChar)) {
    if (lastComma !== -1 && lastComma < str.length) str = str.slice(0, lastComma);
  }

  // Close any still-open brackets in reverse order.
  let out = str.replace(/,\s*$/, "");
  for (let i = stack.length - 1; i >= 0; i--) {
    out += stack[i] === "{" ? "}" : "]";
  }
  return out;
}

const toNum = (v: string | null): number | null => {
  if (v == null) return null;
  const n = parseInt(v, 10);
  return isFinite(n) ? n : null;
};

/** Extract rate-limit info from provider response headers (best-effort). */
export function extractRateLimit(provider: string, headers: Headers): RateLimitInfo | null {
  try {
    if (provider === "groq") {
      return {
        provider,
        requestsRemaining: toNum(headers.get("x-ratelimit-remaining-requests")),
        requestsLimit: toNum(headers.get("x-ratelimit-limit-requests")),
        tokensRemaining: toNum(headers.get("x-ratelimit-remaining-tokens")),
        resetText: headers.get("x-ratelimit-reset-requests"),
      };
    }
    if (provider === "openrouter") {
      return {
        provider,
        requestsRemaining: toNum(headers.get("x-ratelimit-remaining")),
        requestsLimit: toNum(headers.get("x-ratelimit-limit")),
        tokensRemaining: null,
        resetText: headers.get("x-ratelimit-reset"),
      };
    }
  } catch {
    /* noop */
  }
  return null;
}

/**
 * Normalize any robotic "score" to a consistent 0-100 integer scale.
 * Different models/providers sometimes return the score as a fraction
 * (e.g. 0.72 meaning 72%) or already as a percent (72). This caused wildly
 * different displayed values (0.72% vs 80%) depending on the provider.
 */
function normalizeScore(value: unknown): number {
  let n = typeof value === "number" ? value : parseFloat(String(value));
  if (!isFinite(n)) return 0;
  if (n < 0) n = 0;
  // A value in (0,1] almost certainly represents a fraction -> convert to %.
  if (n > 0 && n <= 1) n = n * 100;
  if (n > 100) n = 100;
  return Math.round(n);
}

/**
 * Walk a parsed result object and normalize all score-like fields in place,
 * regardless of which mode produced it.
 */
export function normalizeResult(mode: number, result: any): any {
  if (!result || typeof result !== "object") return result;

  const scoreKeys = ["score", "scoreBefore", "scoreAfter"];
  for (const k of scoreKeys) {
    if (k in result) result[k] = normalizeScore(result[k]);
  }
  if (Array.isArray(result.iterations)) {
    for (const it of result.iterations) {
      if (it && "score" in it) it.score = normalizeScore(it.score);
    }
  }
  if (Array.isArray(result.comparisons)) {
    for (const c of result.comparisons) {
      if (c && "score" in c) c.score = normalizeScore(c.score);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Provider dispatch (OpenAI-compatible: Groq, OpenRouter, Cohere)
// ---------------------------------------------------------------------------

export interface RateLimitInfo {
  provider: string;
  requestsRemaining?: number | null;
  requestsLimit?: number | null;
  tokensRemaining?: number | null;
  resetText?: string | null;
}

export interface DispatchResult {
  ok: boolean;
  status: number;
  json?: any;
  errorCode?: "API_LIMIT_REACHED";
  message?: string;
  rateLimit?: RateLimitInfo | null;
}

export async function runAnalysis(params: {
  mode: number;
  text: string;
  texts?: string[];
  parameters?: any;
  provider: string;
  apiKey?: string;
  model?: string;
}): Promise<DispatchResult> {
  const { mode, text, texts, parameters, provider } = params;
  const responseSchema = SCHEMA_BY_MODE[mode];
  const promptContent = buildPrompt(mode, text, texts, parameters);
  const selectedModel = params.model || getDefaultModel(provider);
  const temperature = mode === 3 || mode === 4 ? 0.2 : 0;

  // Output token budget. Rewrite modes (3, 4) and comparison (5) must be able
  // to return the FULL rewritten text plus JSON overhead, otherwise the answer
  // gets truncated to a fragment. Estimate from input length (~1 token per 3
  // Ukrainian/Russian chars) and give rewrite modes extra headroom.
  const inputLen = mode === 5
    ? (texts || []).reduce((s, t) => s + (t?.length || 0), 0)
    : (text?.length || 0);
  const inputTokens = Math.ceil(inputLen / 3);
  let maxOutputTokens: number;
  if (mode === 3) maxOutputTokens = Math.min(32000, inputTokens * 2 + 2000);
  else if (mode === 4) maxOutputTokens = Math.min(32000, inputTokens * 3 + 3000);
  else if (mode === 5) maxOutputTokens = Math.min(24000, inputTokens + 4000);
  // Mode 2 returns 12 categories + many structural patterns + AI-overview +
  // conclusion. It must scale with input, otherwise long texts truncate the
  // JSON mid-array (the "Expected ',' or ']'" error) and drop markers.
  else if (mode === 2) maxOutputTokens = Math.min(24000, 6000 + inputTokens * 2);
  else maxOutputTokens = Math.min(8000, 2000 + inputTokens);

  // ---- Gemini path ----
  if (provider === "gemini") {
    const apiKey = params.apiKey || process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return { ok: false, status: 400, message: "Вбудований ключ Gemini не знайдено на сервері, і власний ключ не вказано в налаштуваннях." };
    }
    const ai = getAiClient(apiKey);
    const response = await ai.models.generateContent({
      model: selectedModel,
      contents: promptContent,
      config: {
        systemInstruction: SYSTEM_INSTRUCTION,
        temperature,
        maxOutputTokens,
        responseMimeType: "application/json",
        responseSchema,
      },
    });
    const outputText = response.text;
    if (!outputText) throw new Error("Порожня відповідь від моделі Gemini API.");
    return { ok: true, status: 200, json: normalizeResult(mode, parseModelJson(outputText)) };
  }

  // ---- OpenAI-compatible providers ----
  let actualApiKey = params.apiKey || "";
  if (!actualApiKey) {
    if (provider === "groq") actualApiKey = process.env.GROQ_API_KEY || "";
    else if (provider === "openrouter") actualApiKey = process.env.OPENROUTER_API_KEY || "";
    else if (provider === "cohere") actualApiKey = process.env.COHERE_API_KEY || "";
  }
  if (!actualApiKey) {
    return { ok: false, status: 400, message: `Для використання провайдера ${provider.toUpperCase()} необхідно вказати власний API ключ у налаштуваннях ШІ.` };
  }

  let endpoint = "";
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  let requestBody: any = {};
  const userMessage = `${promptContent}\n\n${getSchemaPrompt(responseSchema)}`;

  if (provider === "groq") {
    endpoint = "https://api.groq.com/openai/v1/chat/completions";
    headers["Authorization"] = `Bearer ${actualApiKey}`;
    requestBody = {
      model: selectedModel,
      messages: [
        { role: "system", content: SYSTEM_INSTRUCTION },
        { role: "user", content: userMessage },
      ],
      temperature,
      max_tokens: maxOutputTokens,
      response_format: { type: "json_object" },
    };
  } else if (provider === "openrouter") {
    endpoint = "https://openrouter.ai/api/v1/chat/completions";
    headers["Authorization"] = `Bearer ${actualApiKey}`;
    headers["HTTP-Referer"] = process.env.APP_URL || "https://linguaforensic.vercel.app";
    headers["X-Title"] = "LinguaForensic";
    requestBody = {
      model: selectedModel,
      messages: [
        { role: "system", content: SYSTEM_INSTRUCTION },
        { role: "user", content: userMessage },
      ],
      temperature,
      max_tokens: maxOutputTokens,
      response_format: { type: "json_object" },
    };
  } else if (provider === "cohere") {
    endpoint = "https://api.cohere.com/v2/chat";
    headers["Authorization"] = `Bearer ${actualApiKey}`;
    requestBody = {
      model: selectedModel,
      messages: [
        { role: "system", content: SYSTEM_INSTRUCTION },
        { role: "user", content: userMessage },
      ],
      temperature,
      max_tokens: maxOutputTokens,
      response_format: { type: "json_object" },
    };
  } else {
    return { ok: false, status: 400, message: `Невідомий провайдер: ${provider}.` };
  }

  let apiResponse = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(requestBody),
  });

  // Some models on Groq/OpenRouter reject strict json_object mode with a 400
  // "Failed to validate JSON". Retry once WITHOUT response_format; our tolerant
  // parser then extracts the JSON from the plain-text answer.
  if (!apiResponse.ok && apiResponse.status === 400 && requestBody.response_format) {
    const errPeek = (await apiResponse.clone().text().catch(() => "")).toLowerCase();
    if (errPeek.includes("json") || errPeek.includes("validate") || errPeek.includes("response_format")) {
      const retryBody = { ...requestBody };
      delete retryBody.response_format;
      apiResponse = await fetch(endpoint, {
        method: "POST",
        headers,
        body: JSON.stringify(retryBody),
      });
    }
  }

  if (!apiResponse.ok) {
    const errorText = await apiResponse.text().catch(() => "");
    let parsedErr: any = {};
    try { parsedErr = JSON.parse(errorText); } catch { /* noop */ }
    const errorDetail = parsedErr?.error?.message || parsedErr?.message || errorText || "Unknown error";
    const low = errorDetail.toLowerCase();
    const isLimit =
      apiResponse.status === 429 ||
      apiResponse.status === 402 ||
      low.includes("quota") || low.includes("limit") || low.includes("exhausted") ||
      low.includes("balance") || low.includes("credit");
    if (isLimit) {
      return {
        ok: false,
        status: 429,
        errorCode: "API_LIMIT_REACHED",
        message: `Ліміти провайдера ${provider.toUpperCase()} вичерпані. Змініть постачальника, поповніть баланс або скористайтеся вбудованим Google Gemini.`,
      };
    }
    return { ok: false, status: apiResponse.status, message: `Помилка API провайдера ${provider.toUpperCase()} (${apiResponse.status}): ${errorDetail}` };
  }

  const responseData = await apiResponse.json();
  const rateLimit = extractRateLimit(provider, apiResponse.headers);
  let choiceText = "";
  if (provider === "cohere") {
    choiceText = responseData.message?.content?.[0]?.text || responseData.text || "";
  } else {
    choiceText = responseData.choices?.[0]?.message?.content || "";
  }
  if (!choiceText) throw new Error(`Порожня відповідь від провайдера ${provider}.`);
  return { ok: true, status: 200, json: normalizeResult(mode, parseModelJson(choiceText)), rateLimit };
}
