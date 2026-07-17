import type { VercelRequest, VercelResponse } from "@vercel/node";
import { runAnalysis } from "./_lib.js";

export const config = {
  maxDuration: 60, // seconds — long analyses need headroom (Hobby allows up to 60s)
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method Not Allowed" });
    return;
  }

  try {
    const { mode, text, texts, parameters, apiSettings } = req.body || {};

    if (!mode || (mode !== 5 && !text) || (mode === 5 && (!texts || texts.length < 2))) {
      res.status(400).json({ error: "Некоректні вхідні дані. Потрібен режим і текст." });
      return;
    }

    const provider = apiSettings?.provider || "gemini";
    const userApiKey = apiSettings?.apiKey || "";
    const selectedModel = apiSettings?.model || "";

    try {
      const result = await runAnalysis({
        mode,
        text,
        texts,
        parameters,
        provider,
        apiKey: userApiKey,
        model: selectedModel,
      });

      if (result.ok) {
        res.status(200).json(result.json);
        return;
      }

      if (result.errorCode === "API_LIMIT_REACHED") {
        res.status(429).json({ error: "API_LIMIT_REACHED", message: result.message });
        return;
      }

      res.status(result.status).json({ error: result.message });
    } catch (apiError: any) {
      console.error("[LinguaForensic] API Error:", apiError);
      const errMsg = String(apiError?.message || apiError || "").toLowerCase();
      const isLimit =
        errMsg.includes("quota") || errMsg.includes("limit") || errMsg.includes("exhausted") ||
        errMsg.includes("balance") || errMsg.includes("credit") || errMsg.includes("429") || errMsg.includes("402");
      if (isLimit) {
        res.status(429).json({
          error: "API_LIMIT_REACHED",
          message: "Ліміти або баланс вашого API ключа вичерпані. Змініть постачальника або переключіться на вбудований Google Gemini.",
        });
      } else {
        res.status(500).json({ error: apiError?.message || "Внутрішня помилка сервера під час обробки запиту ШІ." });
      }
    }
  } catch (outerError: any) {
    console.error("[LinguaForensic] Outer Error:", outerError);
    res.status(500).json({ error: "Внутрішня помилка сервера при обробці запиту." });
  }
}
