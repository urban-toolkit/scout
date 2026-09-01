import { appUrl } from "./runtimePaths";

export type AiApiType = "openai_compatible" | "anthropic" | "gemini";

export interface AiSettings {
  apiType: AiApiType;
  baseUrl: string;
  model: string;
  hasApiKey: boolean;
  configured: boolean;
}

export interface AiSettingsPatch {
  apiType?: AiApiType;
  baseUrl?: string;
  model?: string;
  /** Omit or leave blank to keep the currently saved key. */
  apiKey?: string;
  clearApiKey?: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

async function asJson(res: Response) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body?.error || `Request failed (${res.status})`);
  }
  return body;
}

export async function getAiSettings(): Promise<AiSettings> {
  const res = await fetch(appUrl("/api/ai/settings"));
  return asJson(res);
}

export async function saveAiSettings(
  patch: AiSettingsPatch,
): Promise<AiSettings> {
  const res = await fetch(appUrl("/api/ai/settings"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  return asJson(res);
}

export interface ProviderModelsResult {
  models: string[];
  listable: boolean;
}

export async function getProviderModels(input: {
  apiType?: AiApiType;
  baseUrl?: string;
  apiKey?: string;
}): Promise<ProviderModelsResult> {
  const res = await fetch(appUrl("/api/ai/provider-models"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return asJson(res);
}

export async function sendChatMessage(
  messages: ChatMessage[],
): Promise<string> {
  const res = await fetch(appUrl("/api/ai/chat"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  const body = await asJson(res);
  return body.reply as string;
}
