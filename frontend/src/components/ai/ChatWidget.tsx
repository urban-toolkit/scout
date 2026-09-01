import { useEffect, useRef, useState } from "react";
import Fab from "@mui/material/Fab";
import Paper from "@mui/material/Paper";
import Slide from "@mui/material/Slide";
import IconButton from "@mui/material/IconButton";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
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

const PANEL_WIDTH = 400;

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
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
      const reply = await sendChatMessage(nextMessages);
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
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
            <Typography variant="subtitle1" fontWeight={600}>
              AI Chat
            </Typography>
            <Box>
              <IconButton size="small" onClick={() => setSettingsOpen(true)} aria-label="AI settings">
                <SettingsIcon fontSize="small" />
              </IconButton>
              <IconButton size="small" onClick={() => setOpen(false)} aria-label="Close chat">
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
                  Open AI Settings
                </Button>
              </Stack>
            ) : messages.length === 0 ? (
              <Typography variant="body2" color="text.secondary" sx={{ pt: 2, textAlign: "center" }}>
                Ask me anything.
              </Typography>
            ) : (
              <Stack spacing={1}>
                {messages.map((m, i) => (
                  <Box
                    key={i}
                    sx={{
                      alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                      bgcolor: m.role === "user" ? "primary.main" : "#f1f5f9",
                      color: m.role === "user" ? "primary.contrastText" : "text.primary",
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
              <Typography variant="caption" color="error" sx={{ display: "block", mt: 1 }}>
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
              placeholder={configured ? "Message the AI…" : "Set up AI to chat"}
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
