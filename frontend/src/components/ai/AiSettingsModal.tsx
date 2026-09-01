import { useEffect, useState } from "react";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Alert from "@mui/material/Alert";
import Stack from "@mui/material/Stack";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Link from "@mui/material/Link";
import CircularProgress from "@mui/material/CircularProgress";

import {
  AiApiType,
  AiSettings,
  getAiSettings,
  getProviderModels,
  saveAiSettings,
} from "../../utils/aiApi";

type UiMode = "openai" | "anthropic" | "gemini" | "custom";

const PROVIDER_INFO: Record<
  UiMode,
  { apiType: AiApiType; modelHint: string; keyLink?: string; keyLinkLabel?: string }
> = {
  openai: {
    apiType: "openai_compatible",
    modelHint: "gpt-4o-mini",
    keyLink: "https://platform.openai.com/api-keys",
    keyLinkLabel: "Get your OpenAI key",
  },
  anthropic: {
    apiType: "anthropic",
    modelHint: "claude-haiku-4-5-20251001",
    keyLink: "https://console.anthropic.com/keys",
    keyLinkLabel: "Get your Anthropic key",
  },
  gemini: {
    apiType: "gemini",
    modelHint: "gemini-2.0-flash",
    keyLink: "https://aistudio.google.com/apikey",
    keyLinkLabel: "Get your Gemini key",
  },
  custom: {
    apiType: "openai_compatible",
    modelHint: "llama3.2",
  },
};

function uiModeFromSaved(apiType: AiApiType, baseUrl: string): UiMode {
  if (apiType === "anthropic") return "anthropic";
  if (apiType === "gemini") return "gemini";
  if (baseUrl) return "custom";
  return "openai";
}

interface Props {
  open: boolean;
  onClose: () => void;
  onSaved?: (settings: AiSettings) => void;
}

export default function AiSettingsModal({ open, onClose, onSaved }: Props) {
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const [uiMode, setUiMode] = useState<UiMode>("openai");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [hasApiKey, setHasApiKey] = useState(false);

  // What the configured endpoint says it serves. Empty means "we have not
  // asked, or it could not tell us", and the Model field stays free text.
  const [models, setModels] = useState<string[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setError(null);
    setSuccess(false);
    setApiKey("");
    setModels([]);
    setModelsError(null);
    setLoading(true);
    getAiSettings()
      .then((s) => {
        setUiMode(uiModeFromSaved(s.apiType, s.baseUrl));
        setBaseUrl(s.baseUrl || "");
        setModel(s.model || "");
        setHasApiKey(s.hasApiKey);
      })
      .catch((e) => setError(e.message || "Failed to load AI settings."))
      .finally(() => setLoading(false));
  }, [open]);

  const handleModeChange = (_: unknown, next: UiMode | null) => {
    if (!next) return;
    setUiMode(next);
    if (next !== "custom") setBaseUrl("");
    // A list fetched from one provider must not be offered for another.
    setModels([]);
    setModelsError(null);
  };

  const loadModels = async () => {
    setLoadingModels(true);
    setModelsError(null);
    try {
      const res = await getProviderModels({
        apiType: PROVIDER_INFO[uiMode].apiType,
        baseUrl: uiMode === "custom" ? baseUrl : "",
        // Send what is on screen, not what is saved - the point is to pick a
        // model for the endpoint being configured right now. A blank key
        // means "use the saved one", which the server resolves.
        apiKey,
      });
      setModels(res.models || []);
      if (!res.listable) {
        setModelsError("This provider does not publish a model list.");
      } else if (!res.models?.length) {
        setModelsError("The endpoint returned no models.");
      }
    } catch (e: any) {
      setModels([]);
      setModelsError(e?.message || "Could not reach the endpoint.");
    } finally {
      setLoadingModels(false);
    }
  };

  const handleRemoveKey = async () => {
    setSaving(true);
    setError(null);
    try {
      const s = await saveAiSettings({ clearApiKey: true });
      setHasApiKey(s.hasApiKey);
      setApiKey("");
      onSaved?.(s);
    } catch (e: any) {
      setError(e.message || "Failed to remove key.");
    } finally {
      setSaving(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccess(false);
    try {
      const info = PROVIDER_INFO[uiMode];
      const s = await saveAiSettings({
        apiType: info.apiType,
        baseUrl: uiMode === "custom" ? baseUrl : "",
        model,
        apiKey: apiKey || undefined,
      });
      setHasApiKey(s.hasApiKey);
      setApiKey("");
      setSuccess(true);
      onSaved?.(s);
    } catch (e: any) {
      setError(e.message || "Failed to save AI settings.");
    } finally {
      setSaving(false);
    }
  };

  const info = PROVIDER_INFO[uiMode];

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>AI Settings</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Configure the provider that powers the LLM chat. Your API key is
            stored on the SCOUT backend, not in the browser.
          </Typography>

          <ToggleButtonGroup
            value={uiMode}
            exclusive
            onChange={handleModeChange}
            size="small"
            disabled={loading}
          >
            <ToggleButton value="openai">OpenAI</ToggleButton>
            <ToggleButton value="anthropic">Anthropic</ToggleButton>
            <ToggleButton value="gemini">Gemini</ToggleButton>
            <ToggleButton value="custom">Custom</ToggleButton>
          </ToggleButtonGroup>

          {uiMode === "custom" && (
            <TextField
              label="Base URL"
              placeholder="http://localhost:11434/v1  (Ollama, LM Studio, vLLM, …)"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              helperText="Any OpenAI-compatible endpoint."
              fullWidth
              size="small"
              disabled={loading}
            />
          )}

          <TextField
            label="API Key"
            type="password"
            autoComplete="new-password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={hasApiKey ? "••••••••  (saved — leave blank to keep)" : "Enter your API key"}
            helperText={
              info.keyLink ? (
                <Link href={info.keyLink} target="_blank" rel="noreferrer">
                  {info.keyLinkLabel}
                </Link>
              ) : (
                "Optional for keyless local servers."
              )
            }
            fullWidth
            size="small"
            disabled={loading}
          />
          {hasApiKey && (
            <Button
              size="small"
              color="error"
              variant="text"
              sx={{ alignSelf: "flex-start", mt: "-8px !important" }}
              onClick={handleRemoveKey}
              disabled={saving || loading}
            >
              Remove saved key
            </Button>
          )}

          {models.length > 0 ? (
            <TextField
              select
              label="Model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              fullWidth
              size="small"
              disabled={loading}
            >
              <MenuItem value="">
                <em>Select a model…</em>
              </MenuItem>
              {models.map((m) => (
                <MenuItem key={m} value={m}>
                  {m}
                </MenuItem>
              ))}
              {/* A model saved earlier that the endpoint no longer lists
                  would otherwise vanish from the box that claims to show it. */}
              {model && !models.includes(model) && (
                <MenuItem value={model}>{model} (not listed)</MenuItem>
              )}
            </TextField>
          ) : (
            <TextField
              label="Model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder={info.modelHint}
              fullWidth
              size="small"
              disabled={loading}
            />
          )}

          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: "-8px !important" }}>
            <Button
              size="small"
              variant="text"
              onClick={() => void loadModels()}
              disabled={loading || loadingModels}
              startIcon={loadingModels ? <CircularProgress size={14} /> : undefined}
            >
              {loadingModels
                ? "Fetching models…"
                : models.length > 0
                  ? "Refresh models"
                  : "Fetch models"}
            </Button>
            {modelsError && (
              <Typography variant="caption" color="error">
                {modelsError}
              </Typography>
            )}
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ mt: "-8px !important" }}>
            Asks the endpoint above what it serves.
          </Typography>

          {error && <Alert severity="error">{error}</Alert>}
          {success && <Alert severity="success">Settings saved.</Alert>}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={saving}>
          Close
        </Button>
        <Button onClick={handleSave} variant="contained" disabled={saving || loading}>
          {saving ? "Saving…" : "Save"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
