import type { Plugin } from "@opencode-ai/plugin"
import { readdir } from "fs/promises"
import { join } from "path"

/**
 * Watches for new dictation files and injects a notification into the active session.
 *
 * When a new .md file appears in dictations/, this plugin sends a noReply message
 * into the current session so the agent sees it on its next turn. The agent cannot
 * miss it — it appears as a user message in the conversation.
 */
export const DictationNotifier: Plugin = async ({ client, directory }) => {
  const dictationsDir = join(directory, "dictations")
  const knownFiles = new Set<string>()

  // Seed with existing files so we only notify on genuinely new ones
  try {
    for (const f of await readdir(dictationsDir)) {
      if (f.endsWith(".md")) knownFiles.add(f)
    }
  } catch {
    // dictations/ doesn't exist yet — that's fine
  }

  return {
    "file.watcher.updated": async (input) => {
      // Check if the update is a new dictation file
      const path: string = (input as any).path ?? ""
      if (!path.includes("dictations") || !path.endsWith(".md")) return

      const filename = path.split(/[\\/]/).pop()!
      if (knownFiles.has(filename)) return
      knownFiles.add(filename)

      // Find the active session to inject into
      const sessions = await client.session.list()
      if (!sessions.data || sessions.data.length === 0) return

      // Inject into all active sessions — the agent will see it on next turn
      for (const session of sessions.data) {
        try {
          await client.session.prompt({
            path: { id: session.id },
            body: {
              noReply: true,
              parts: [{
                type: "text",
                text: `[DICTATION NOTIFICATION] New dictation file: dictations/${filename} — Max has written new instructions. Read it immediately before continuing other work.`,
              }],
            },
          })
        } catch {
          // Session might not be active — that's fine
        }
      }
    },
  }
}
