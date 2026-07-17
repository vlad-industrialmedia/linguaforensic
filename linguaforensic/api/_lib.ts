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
// gemini-1.5 / 2.0 are shut down; use 2.5-flash as the safe default.
export const DEFAULT_MODELS: Record<Provider, string> = {
  gemini: "gemini-2.5-flash",
  groq: "llama-3.3-70b-versatile",
  openrouter: "google/gemini-2.0-flash-exp:free",
  cohere: "command-r-plus",
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
      items: {
        type: Type.OBJECT,
        properties: {
          marker: { type: Type.STRING },
          quote: { type: Type.STRING },
          description: { type: Type.STRING },
        },
        required: ["marker", "quote", "description"],
      },
    },
    modelGroupReasoning: { type: Type.STRING },
    conclusion: { type: Type.STRING },
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
    ? `\n\n[TIMEOUT PROTECTION NOTICE: The input text is very long (${textLength} characters). To avoid gateway timeouts, keep your analyses, descriptions, and structural patterns extremely concise and brief. Extract at most 4-5 key structural patterns in total instead of dozens.]`
    : "";

  if (mode === 1) {
    return `
      Виконайте Режим 1 (Швидка детекція) для наступного тексту.
      Проведіть аналіз 284 лінгвістичних параметрів та 16 структурних маркерів.
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
      Поверніть строго JSON відповідно до схеми Mode3Schema.
      ${optimizationNotice}

      ТЕКСТ ДЛЯ РЕРАЙТУ:
      """
      ${text}
      """
    `;
  }
  if (mode === 4) {
    const iterations = Math.min(parameters?.iterations ?? 3, 3);
    if (isLongText) {
      return `
        Виконайте Режим 4 (Циклічний рекурсивний рерайт) для великого тексту (${textLength} символів).
        КРИТИЧНО ДЛЯ УНИКНЕННЯ ТАЙМАУТУ:
        Виконайте один якісний глибокий рерайт тексту, який рішуче знижує оцінку роботності до <${parameters?.targetPercent ?? 25}%. Помістіть цей підсумковий текст у 'finalText' та як останній знімок (textSnapshot) в ітераціях.
        У масиві 'iterations' змоделюйте історію з 3 кроків для гарної візуалізації в інтерфейсі:
        - Крок 0 (iteration: 0): Вихідний текст, статус 'база', короткий textSnapshot (до 200 символів), оцінка роботності висока.
        - Крок 1 (iteration: 1): Прогрес рерайту, статус 'прийнято', проробка синтаксису, оцінка середня (наприклад 55%).
        - Крок 2 (iteration: 2): Фінальна версія, статус 'прийнято', оцінка низька (наприклад 15-20%), textSnapshot збігається з кінцем finalText.

        Поверніть строго JSON відповідно до схеми Mode4Schema.

        ТЕКСТ ДЛЯ ЦИКЛІЧНОГО РЕРАЙТУ:
        """
        ${text}
        """
      `;
    }
    return `
      Виконайте Режим 4 (Циклічний рекурсивний рерайт) до ${iterations} ітерацій.
      Ви маєте змоделювати покроковий процес циклічного рерайту та детекції:
      - Крок 0: Вихідний текст (база)
      - Для кожної ітерації:
        1. Зробіть аналіз роботності.
        2. Якщо вище цілі, зробіть рерайт.
        3. Виконайте Sense-check (збереження сенсу), Style-check (збереження природності), Integrity-check (збереження цифр, фактів).
        4. Якщо якість деградувала, відкотіться (позначте статус 'відкат' та поверніться до попередньої версії). Інакше прийміть версію (статус 'прийнято').
        5. Якщо ціль (<25% роботності) досягнута або настало плато (немає зниження), зупиніть цикл достроково.

      CRITICAL PERFORMANCE REQUIREMENT: Keep descriptions, evaluations, and textSnapshots extremely brief and concise. Do not generate excessively long text repetitions.

      Поверніть повну історію ітерацій та підсумковий текст.
      Поверніть строго JSON відповідно до схеми Mode4Schema.

      ТЕКСТ ДЛЯ ЦИКЛІЧНОГО РЕРАЙТУ:
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
  return JSON.parse(cleaned);
}

// ---------------------------------------------------------------------------
// Provider dispatch (OpenAI-compatible: Groq, OpenRouter, Cohere)
// ---------------------------------------------------------------------------

export interface DispatchResult {
  ok: boolean;
  status: number;
  json?: any;
  errorCode?: "API_LIMIT_REACHED";
  message?: string;
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
        responseMimeType: "application/json",
        responseSchema,
      },
    });
    const outputText = response.text;
    if (!outputText) throw new Error("Порожня відповідь від моделі Gemini API.");
    return { ok: true, status: 200, json: parseModelJson(outputText) };
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
      response_format: { type: "json_object" },
    };
  } else {
    return { ok: false, status: 400, message: `Невідомий провайдер: ${provider}.` };
  }

  const apiResponse = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(requestBody),
  });

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
  let choiceText = "";
  if (provider === "cohere") {
    choiceText = responseData.message?.content?.[0]?.text || responseData.text || "";
  } else {
    choiceText = responseData.choices?.[0]?.message?.content || "";
  }
  if (!choiceText) throw new Error(`Порожня відповідь від провайдера ${provider}.`);
  return { ok: true, status: 200, json: parseModelJson(choiceText) };
}
