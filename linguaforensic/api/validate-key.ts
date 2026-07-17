import type { VercelRequest, VercelResponse } from "@vercel/node";
import { getAiClient, getDefaultModel } from "./_lib.js";

export const config = {
  maxDuration: 30,
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ valid: false, error: "Method Not Allowed" });
    return;
  }

  try {
    const { provider, apiKey, model } = req.body || {};
    if (!apiKey) {
      res.status(400).json({ valid: false, error: "Будь ласка, введіть API ключ для перевірки." });
      return;
    }

    if (provider === "gemini") {
      try {
        const ai = getAiClient(apiKey);
        await ai.models.generateContent({
          model: model || getDefaultModel("gemini"),
          contents: "ping",
          config: { maxOutputTokens: 5 },
        });
        res.json({ valid: true, message: "Ключ Gemini успішно активовано та перевірено!" });
      } catch (err: any) {
        res.json({ valid: false, error: `Помилка Gemini API: ${err?.message || err}` });
      }
      return;
    }

    const openAiCompatible: Record<string, { url: string; extraHeaders?: Record<string, string> }> = {
      groq: { url: "https://api.groq.com/openai/v1/chat/completions" },
      openrouter: {
        url: "https://openrouter.ai/api/v1/chat/completions",
        extraHeaders: {
          "HTTP-Referer": process.env.APP_URL || "https://linguaforensic.vercel.app",
          "X-Title": "LinguaForensic",
        },
      },
      cohere: { url: "https://api.cohere.com/v2/chat" },
    };

    const cfg = openAiCompatible[provider];
    if (!cfg) {
      res.status(400).json({ valid: false, error: "Невідомий провайдер для валідації." });
      return;
    }

    try {
      const apiResponse = await fetch(cfg.url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
          ...(cfg.extraHeaders || {}),
        },
        body: JSON.stringify({
          model: model || getDefaultModel(provider),
          messages: [{ role: "user", content: "ping" }],
          max_tokens: 5,
        }),
      });
      if (apiResponse.ok) {
        res.json({ valid: true, message: `Ключ ${provider.toUpperCase()} успішно активовано та перевірено!` });
      } else {
        const errText = await apiResponse.text();
        let parsedErr: any = {};
        try { parsedErr = JSON.parse(errText); } catch { /* noop */ }
        const detail = parsedErr?.error?.message || parsedErr?.message || errText || `HTTP ${apiResponse.status}`;
        res.json({ valid: false, error: `Помилка ${provider.toUpperCase()} API (${apiResponse.status}): ${detail}` });
      }
    } catch (err: any) {
      res.json({ valid: false, error: `Помилка з'єднання з ${provider.toUpperCase()}: ${err?.message || err}` });
    }
  } catch (err: any) {
    console.error("[LinguaForensic] Validation Error:", err);
    res.status(500).json({ valid: false, error: `Внутрішня помилка перевірки: ${err?.message || err}` });
  }
}
