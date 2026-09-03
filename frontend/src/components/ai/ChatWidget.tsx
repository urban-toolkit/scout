import { useEffect, useRef, useState } from "react";
import { useReactFlow } from "@xyflow/react";
import Fab from "@mui/material/Fab";
import Paper from "@mui/material/Paper";
import Slide from "@mui/material/Slide";
import IconButton from "@mui/material/IconButton";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import SmartToyOutlinedIcon from "@mui/icons-material/SmartToyOutlined";
import CloseIcon from "@mui/icons-material/Close";
import SettingsIcon from "@mui/icons-material/Settings";
import SendIcon from "@mui/icons-material/Send";

import AiSettingsModal from "./AiSettingsModal";
import {
  AiSettings,
  ChatMessage,
  getAiSettings,
  sendChatMessage,
} from "../../utils/aiApi";
import {
  applyAgentAction,
  buildSystemPrompt,
  collectWidgetSummaries,
  parseAgentReply,
  validateAction,
} from "../../agents/widgetAgent";
import {
  applyAgentAction as applyDataAgentAction,
  buildSystemPrompt as buildDataAgentSystemPrompt,
  collectPyCodeNodeSummaries,
  parseAgentReply as parseDataAgentReply,
  validateAction as validateDataAgentAction,
} from "../../agents/dataAgent";
import {
  applyAgentAction as applyNodeAgentAction,
  buildSystemPrompt as buildNodeAgentSystemPrompt,
  collectNodeSummaries,
  parseAgentReply as parseNodeAgentReply,
  validateAction as validateNodeAgentAction,
} from "../../agents/nodeAgent";

const PANEL_WIDTH = 400;

type AgentMode = "general" | "widget" | "data" | "node";

const AGENT_LABELS: Record<AgentMode, string> = {
  general: "General Chat",
  widget: "Widget Agent",
  data: "Data Agent",
  node: "Node Agent",
};

const AGENT_EMPTY_HINTS: Record<AgentMode, string> = {
  general: "Ask me anything.",
  widget: "Tell me which widget to change, e.g. “set season to winter”.",
  data: "Tell me what Python code to write, e.g. “in Node A, load the CSV and print its columns”.",
  node: "Tell me what node to add or rename, e.g. “add a data layer node named City Buildings”.",
};

export default function ChatWidget() {
  const rf = useReactFlow();
  const [open, setOpen] = useState(false);
  const [agentMode, setAgentMode] = useState<AgentMode>("general");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [aiSettings, setAiSettings] = useState<AiSettings | null>(null);
  const [checkingSettings, setCheckingSettings] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement | null>(null);

  const refreshSettings = () => {
    setCheckingSettings(true);
    getAiSettings()
      .then(setAiSettings)
      .catch(() => setAiSettings(null))
      .finally(() => setCheckingSettings(false));
  };

  useEffect(() => {
    if (open) refreshSettings();
  }, [open]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, sending]);

  const handleAgentChange = (next: AgentMode) => {
    if (next === agentMode) return;
    setAgentMode(next);
    // A widget's system prompt (and a general reply) from one agent would be
    // a confusing carry-over into the other, so each switch starts fresh.
    setMessages([]);
    setError(null);
  };

  const sendGeneralMessage = async (nextMessages: ChatMessage[]) => {
    const reply = await sendChatMessage(nextMessages);
    setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
  };

  const sendWidgetAgentMessage = async (nextMessages: ChatMessage[]) => {
    const widgets = collectWidgetSummaries(rf.getNodes());
    if (!widgets.length) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "There are no widgets on the canvas right now, so there's nothing I can change.",
        },
      ]);
      return;
    }

    const raw = await sendChatMessage([
      { role: "system", content: buildSystemPrompt(widgets) },
      ...nextMessages.filter((m) => m.role !== "system"),
    ]);
    const reply = parseAgentReply(raw);
    const validated = validateAction(reply.action, widgets);

    let content = reply.message;
    if (reply.action && !validated) {
      content +=
        "\n\n(That didn't match a widget/value actually on the canvas, so nothing changed.)";
    } else if (validated) {
      const { ranNodeIds } = applyAgentAction(validated, rf);
      content += ranNodeIds.length
        ? `\n\n(Updated the widget and re-ran ${ranNodeIds.length} connected node${ranNodeIds.length === 1 ? "" : "s"}.)`
        : "\n\n(Updated the widget - it isn't connected to any Code node to re-run.)";
    }

    setMessages((prev) => [...prev, { role: "assistant", content }]);
  };

  const sendDataAgentMessage = async (nextMessages: ChatMessage[]) => {
    const pyNodes = collectPyCodeNodeSummaries(rf.getNodes());
    if (!pyNodes.length) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "There are no Code nodes on the canvas right now, so there's nothing I can write to.",
        },
      ]);
      return;
    }

    const raw = await sendChatMessage([
      { role: "system", content: buildDataAgentSystemPrompt(pyNodes) },
      ...nextMessages.filter((m) => m.role !== "system"),
    ]);
    const reply = parseDataAgentReply(raw);
    const validated = validateDataAgentAction(reply.action, pyNodes);

    let content = reply.message;
    if (reply.action && !validated) {
      content +=
        "\n\n(That didn't name a Code node actually on the canvas, so nothing changed.)";
    } else if (validated) {
      applyDataAgentAction(validated, rf);
      content += "\n\n(Wrote the code and ran the node.)";
    }

    setMessages((prev) => [...prev, { role: "assistant", content }]);
  };

  const sendNodeAgentMessage = async (nextMessages: ChatMessage[]) => {
    // Unlike the Widget/Data agents, there's no "nothing to act on" bail-out
    // here - adding a node is always possible even on an empty canvas.
    const nodes = collectNodeSummaries(rf.getNodes());

    const raw = await sendChatMessage([
      { role: "system", content: buildNodeAgentSystemPrompt(nodes) },
      ...nextMessages.filter((m) => m.role !== "system"),
    ]);
    const reply = parseNodeAgentReply(raw);
    const validated = validateNodeAgentAction(reply.action, nodes);

    let content = reply.message;
    if (reply.action && !validated) {
      content +=
        "\n\n(That didn't name a valid node kind, or a node actually on the canvas, so nothing changed.)";
    } else if (validated) {
      applyNodeAgentAction(validated, rf);
      content += validated.kind === "add" ? "\n\n(Added the node.)" : "\n\n(Renamed the node.)";
    }

    setMessages((prev) => [...prev, { role: "assistant", content }]);
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;

    const nextMessages: ChatMessage[] = [
      ...messages,
      { role: "user", content: text },
    ];
    setMessages(nextMessages);
    setInput("");
    setError(null);
    setSending(true);
    try {
      if (agentMode === "widget") {
        await sendWidgetAgentMessage(nextMessages);
      } else if (agentMode === "data") {
        await sendDataAgentMessage(nextMessages);
      } else if (agentMode === "node") {
        await sendNodeAgentMessage(nextMessages);
      } else {
        await sendGeneralMessage(nextMessages);
      }
    } catch (e: any) {
      setError(e.message || "Failed to get a reply.");
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  const configured = aiSettings?.configured ?? false;

  return (
    <>
      {!open && (
        <Fab
          onClick={() => setOpen(true)}
          sx={{
            position: "fixed",
            right: 24,
            bottom: 24,
            zIndex: 40,
            bgcolor: "grey.700",
            color: "#fff",
            "&:hover": { bgcolor: "grey.800" },
          }}
          aria-label="Open LLM chat"
        >
          <SmartToyOutlinedIcon />
        </Fab>
      )}

      <Slide direction="left" in={open} mountOnEnter unmountOnExit>
        <Paper
          elevation={8}
          square
          sx={{
            position: "fixed",
            top: 0,
            right: 0,
            bottom: 0,
            width: { xs: "100%", sm: PANEL_WIDTH },
            zIndex: 50,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            borderLeft: "1px solid #e5e7eb",
          }}
        >
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              px: 2,
              py: 1.5,
              borderBottom: "1px solid #e5e7eb",
              flexShrink: 0,
            }}
          >
            <Select
              value={agentMode}
              onChange={(e) => handleAgentChange(e.target.value as AgentMode)}
              size="small"
              variant="standard"
              disableUnderline
              disabled={sending}
              sx={{ fontWeight: 600, fontSize: 15 }}
            >
              {(Object.keys(AGENT_LABELS) as AgentMode[]).map((mode) => (
                <MenuItem key={mode} value={mode}>
                  {AGENT_LABELS[mode]}
                </MenuItem>
              ))}
            </Select>
            <Box>
              <IconButton
                size="small"
                onClick={() => setSettingsOpen(true)}
                aria-label="AI settings"
              >
                <SettingsIcon fontSize="small" />
              </IconButton>
              <IconButton
                size="small"
                onClick={() => setOpen(false)}
                aria-label="Close chat"
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            </Box>
          </Box>

          <Box ref={scrollRef} sx={{ flex: 1, overflowY: "auto", p: 2 }}>
            {checkingSettings ? (
              <Box sx={{ display: "flex", justifyContent: "center", pt: 4 }}>
                <CircularProgress size={22} />
              </Box>
            ) : !configured ? (
              <Stack spacing={1.5} sx={{ pt: 2, textAlign: "center" }}>
                <Typography variant="body2" color="text.secondary">
                  AI isn't set up yet. Add a provider, API key and model to
                  start chatting.
                </Typography>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<SettingsIcon />}
                  onClick={() => setSettingsOpen(true)}
                  sx={{ alignSelf: "center" }}
                >
                  AI Settings
                </Button>
              </Stack>
            ) : messages.length === 0 ? (
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ pt: 2, textAlign: "center" }}
              >
                {AGENT_EMPTY_HINTS[agentMode]}
              </Typography>
            ) : (
              <Stack spacing={1}>
                {messages.map((m, i) => (
                  <Box
                    key={i}
                    sx={{
                      alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                      bgcolor: m.role === "user" ? "primary.main" : "#f1f5f9",
                      color:
                        m.role === "user"
                          ? "primary.contrastText"
                          : "text.primary",
                      borderRadius: 2,
                      px: 1.5,
                      py: 0.75,
                      maxWidth: "85%",
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                    }}
                  >
                    <Typography variant="body2">{m.content}</Typography>
                  </Box>
                ))}
                {sending && (
                  <Box sx={{ alignSelf: "flex-start", px: 1.5, py: 0.75 }}>
                    <CircularProgress size={16} />
                  </Box>
                )}
              </Stack>
            )}
            {error && (
              <Typography
                variant="caption"
                color="error"
                sx={{ display: "block", mt: 1 }}
              >
                {error}
              </Typography>
            )}
          </Box>

          <Box
            sx={{
              display: "flex",
              gap: 1,
              p: 1.5,
              borderTop: "1px solid #e5e7eb",
              flexShrink: 0,
            }}
          >
            <TextField
              size="small"
              fullWidth
              placeholder={
                configured
                  ? agentMode === "widget"
                    ? "Tell the agent what to change…"
                    : agentMode === "data"
                      ? "Tell the agent what code to write…"
                      : agentMode === "node"
                        ? "Tell the agent what node to add or rename…"
                        : "Message the AI…"
                  : "Set up AI to chat"
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!configured || sending}
              multiline
              maxRows={4}
            />
            <IconButton
              color="primary"
              onClick={() => void handleSend()}
              disabled={!configured || sending || !input.trim()}
              aria-label="Send"
            >
              <SendIcon />
            </IconButton>
          </Box>
        </Paper>
      </Slide>

      <AiSettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSaved={(s) => setAiSettings(s)}
      />
    </>
  );
}
