# LinguaForensic

Експертна система детекції та корекції AI-тексту з глибокою аналітикою та рерайтом (українською мовою).

Frontend: **React 19 + Vite + Tailwind 4**. Backend: **Vercel Serverless Functions** (`/api`).
Підтримувані провайдери: **Google Gemini** (за замовчуванням), **Groq**, **OpenRouter**, **Cohere**.

## Структура

```
├── api/                  # Serverless-функції (виконуються на боці Vercel)
│   ├── _lib.ts           # Спільна логіка: system prompt, схеми, диспетчер провайдерів
│   ├── analyze.ts        # POST /api/analyze
│   └── validate-key.ts   # POST /api/validate-key
├── src/                  # Frontend (React)
│   ├── App.tsx
│   ├── main.tsx
│   ├── types.ts
│   └── index.css
├── index.html
├── vite.config.ts
├── vercel.json
└── package.json
```

Ключі API ніколи не потрапляють у бандл фронтенду — усі виклики моделей ідуть через `/api`,
а ключі зберігаються в змінних оточення Vercel (або опційно вводяться користувачем в UI і
проксуються сервером).

## Локальний запуск

Потрібен Node.js 18+.

```bash
npm install
```

Створіть файл `.env` (скопіюйте з `.env.example`) і додайте свій `GEMINI_API_KEY`.

Оскільки бекенд — це Vercel-функції, найзручніше запускати через Vercel CLI:

```bash
npm i -g vercel
vercel dev
```

`vercel dev` підніме і фронтенд (Vite), і `/api`-функції на одному порту, тож fetch на
`/api/analyze` працюватиме без окремого проксі.

> Якщо запускати лише `npm run dev` (чистий Vite), працюватиме інтерфейс, але виклики
> `/api/*` будуть недоступні — для повноцінної розробки використовуйте `vercel dev`.

## Деплой на Vercel через GitHub

1. Створіть репозиторій на GitHub і залийте цей код:

   ```bash
   git init
   git add .
   git commit -m "LinguaForensic: Vercel-ready"
   git branch -M main
   git remote add origin https://github.com/<ваш-акаунт>/<репо>.git
   git push -u origin main
   ```

2. На [vercel.com](https://vercel.com) → **Add New… → Project** → імпортуйте цей репозиторій.

3. Vercel сам визначить фреймворк (Vite). Налаштування за замовчуванням підходять:
   - Build Command: `npm run build`
   - Output Directory: `dist`
   - Root Directory: `/` (корінь репозиторію)

4. У **Environment Variables** додайте:
   - `GEMINI_API_KEY` — обов'язково (щоб працював безкоштовний вбудований провайдер);
   - `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `COHERE_API_KEY` — опційно;
   - `APP_URL` — опційно (URL вашого деплою).

5. **Deploy**. Далі кожен `git push` у `main` автоматично оновлюватиме продакшн.

## Що змінено порівняно з версією AI Studio

- Монолітний Express-сервер (`server.ts`) замінено на нативні Vercel-функції в `/api` —
  так надійніше й швидше на serverless.
- Виправлено назви моделей на актуальні (станом на 2026):
  Gemini `1.5/2.0` вимкнено Google → за замовчуванням `gemini-2.5-flash`;
  Groq — `llama-3.3-70b-versatile` (замість неіснуючої `-specdec`).
- Прибрано «провайдер» **Zilliz**: це векторна БД, а не LLM; у старому коді він приховано
  перенаправляв на Gemini, що вводило в оману.
- Прибрано службові Python-скрипти та артефакти збірки з репозиторію.

## Примітка про таймаути

На безкоштовному тарифі Vercel (Hobby) ліміт виконання функції — 60 с. Для дуже великих
текстів у режимах 2/4 передбачено «timeout protection» (модель просять давати стисліші
відповіді). Якщо все одно впираєтесь у ліміт — використовуйте швидкий провайдер (Groq) або
власний ключ Gemini.
