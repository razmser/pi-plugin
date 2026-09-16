import { getModel } from "@earendil-works/pi-ai";
import type { Api, Model } from "@earendil-works/pi-ai";
import type {
  ExtensionAPI,
  ProviderModelConfig,
} from "@earendil-works/pi-coding-agent";

const PROVIDER_ID = "coreinfra";
const PROVIDER_NAME = "CoreInfra AI Hub";
const DEFAULT_HUB_BASE_URL = "https://hub.coreinfra.ai";
const COREINFRA_API_KEY = "X-CoreInfra-Api-Key";
const FETCH_TIMEOUT_MS = 10_000;
type CoreInfraFamily =
  | "openai"
  | "anthropic"
  | "deepseek"
  | "zai"
  | "moonshotai";
const COREINFRA_FAMILIES = [
  "openai",
  "anthropic",
  "deepseek",
  "zai",
  "moonshotai",
] as const;

type ExtraModel = {
  id: string;
  name: string;
  reasoning: boolean;
  thinkingLevelMap?: Model<Api>["thinkingLevelMap"];
  input: ("text" | "image")[];
  contextWindow: number;
  maxTokens: number;
  compat?: ProviderModelConfig["compat"];
};

const EXTRA_MODELS: Partial<Record<CoreInfraFamily, ExtraModel[]>> = {
  zai: [
    {
      id: "glm-4.7-flash",
      name: "GLM-4.7-Flash",
      reasoning: true,
      input: ["text"],
      contextWindow: 200000,
      maxTokens: 131072,
      compat: {
        supportsDeveloperRole: false,
        thinkingFormat: "zai",
        zaiToolStream: true,
      },
    },
  ],
};

type CoreInfraPrices = {
  input_tokens?: number;
  output_tokens?: number;
  cache_read_tokens?: number;
  cache_5m_write_tokens?: number;
  cache_1h_write_tokens?: number;
};

type HubResponse = {
  providers?: Partial<
    Record<
      CoreInfraFamily,
      { models?: Record<string, { display_name?: string; prices?: CoreInfraPrices }> }
    >
  >;
};

function hubBaseUrl(): string {
  return (process.env.COREINFRA_HUB_BASE_URL ?? DEFAULT_HUB_BASE_URL).replace(
    /\/+$/,
    "",
  );
}

function openAiBaseUrl(): string {
  return `${hubBaseUrl()}/openai/api/v1`;
}

function anthropicBaseUrl(): string {
  return `${hubBaseUrl()}/anthropic/api`;
}

function hubApiToken(
  env: NodeJS.ProcessEnv = process.env,
): string | undefined {
  const token = env.COREINFRA_API_KEY?.trim();
  return token ? token : undefined;
}

function pricesRequestInit(token?: string): RequestInit {
  return {
    headers: token ? { [COREINFRA_API_KEY]: token } : {},
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  };
}

async function fetchHubModels(
  fetchImpl: typeof fetch = fetch,
  env: NodeJS.ProcessEnv = process.env,
): Promise<HubResponse> {
  const url = `${hubBaseUrl()}/hub/api/prices`;
  const token = hubApiToken(env);
  const res = await fetchImpl(url, pricesRequestInit(token));

  if (!res.ok) {
    throw new Error(
      `Failed to fetch CoreInfra models: ${res.status} ${res.statusText}`,
    );
  }

  return (await res.json()) as HubResponse;
}

function modelCost(prices: CoreInfraPrices = {}): ProviderModelConfig["cost"] {
  return {
    input: prices.input_tokens ?? 0,
    output: prices.output_tokens ?? 0,
    cacheRead: prices.cache_read_tokens ?? 0,
    cacheWrite: prices.cache_5m_write_tokens ?? prices.cache_1h_write_tokens ?? 0,
  };
}

function builtinModel(
  family: CoreInfraFamily,
  modelId: string,
): Model<Api> | undefined {
  return getModel(family, modelId as never) as Model<Api> | undefined;
}

function familyConfig(family: CoreInfraFamily): {
  api: "openai-responses" | "openai-completions" | "anthropic-messages";
  baseUrl: string;
  compat?: ProviderModelConfig["compat"];
} {
  if (family === "anthropic") {
    return { api: "anthropic-messages", baseUrl: anthropicBaseUrl() };
  }

  if (family === "deepseek") {
    return {
      api: "anthropic-messages",
      baseUrl: anthropicBaseUrl(),
      compat: {
        supportsEagerToolInputStreaming: false,
        forceAdaptiveThinking: true,
      },
    };
  }

  if (family === "moonshotai") {
    return { api: "openai-completions", baseUrl: openAiBaseUrl() };
  }

  // GLM rides its native chat-completions protocol through the hub's OpenAI
  // endpoint, so it inherits pi's built-in `zai` compat (thinking format and
  // tool-streaming) rather than defining an explicit override — the opposite
  // of DeepSeek, which keeps its override for the non-native Anthropic shim.
  if (family === "zai") {
    return { api: "openai-completions", baseUrl: openAiBaseUrl() };
  }

  return { api: "openai-responses", baseUrl: openAiBaseUrl() };
}

function buildModels(hub: HubResponse): {
  models: ProviderModelConfig[];
  warnings: string[];
} {
  const models: ProviderModelConfig[] = [];
  const warnings: string[] = [];

  for (const family of COREINFRA_FAMILIES) {
    const hubModels = hub.providers?.[family]?.models ?? {};
    const { api, baseUrl, compat } = familyConfig(family);

    for (const [modelId, hubModel] of Object.entries(hubModels)) {
      const source =
        builtinModel(family, modelId) ??
        EXTRA_MODELS[family]?.find((m) => m.id === modelId);
      if (!source) {
        warnings.push(`${family}/${modelId} is not known to pi; skipping`);
        continue;
      }

      models.push({
        id: modelId,
        name: hubModel.display_name ?? source.name,
        api,
        baseUrl,
        reasoning: source.reasoning,
        thinkingLevelMap: source.thinkingLevelMap,
        input: source.input,
        cost: modelCost(hubModel.prices),
        contextWindow: source.contextWindow,
        maxTokens: source.maxTokens,
        compat: compat ?? source.compat,
      });
    }
  }

  return { models, warnings };
}

export default async function coreInfraPiPlugin(pi: ExtensionAPI) {
  const { models, warnings } = buildModels(await fetchHubModels());

  pi.registerProvider(PROVIDER_ID, {
    name: PROVIDER_NAME,
    baseUrl: openAiBaseUrl(),
    apiKey: "$COREINFRA_API_KEY",
    api: "openai-responses",
    models,
  });

  for (const warning of warnings) {
    console.warn(`[${PROVIDER_ID}] ${warning}`);
  }
}
