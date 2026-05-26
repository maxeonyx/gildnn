import type { Plugin } from "@opencode-ai/plugin"
import { readdir } from "fs/promises"
import { join } from "path"

/**
 * Watches for new dictation files and injects a notification into the active session.
 *
 * When a new .md file appears in dictations/, this plugin sends a noReply message
 * into the current session so the agent sees it on its next turn.
 *
 * NOTE: Do NOT try to use the file.watcher.updated event for this. The Windows file
 * watcher never fires for externally-created files (tested 2026-05-26, 5 runs, 0 fires).
 * The event hook is correctly wired (generic `event` handler, not a named key) but the
 * underlying OS watcher simply doesn't emit. Use tool.execute.after polling instead.
 */
export const DictationNotifier: Plugin = async ({ client, directory }) => {
  const dictationsDir = join(directory, "dictations")
  const knownFiles = new Set<string>()
  let notifying = false

  // Seed with existing files so we only notify on genuinely new ones
  try {
    for (const f of await readdir(dictationsDir)) {
      if (f.endsWith(".md")) knownFiles.add(f)
    }
  } catch {
    // dictations/ doesn't exist yet — that's fine
  }

  async function checkForNewDictations(sessionID?: string) {
    if (notifying) return
    let newFiles: string[] = []
    try {
      const files = await readdir(dictationsDir)
      for (const f of files) {
        if (f.endsWith(".md") && !knownFiles.has(f)) {
          knownFiles.add(f)
          newFiles.push(f)
        }
      }
    } catch {
      return
    }
    if (newFiles.length === 0) return

    notifying = true
    try {
      const sessions = await client.session.list()
      if (!sessions.data || sessions.data.length === 0) return

      const targets = sessionID
        ? sessions.data.filter(s => s.id === sessionID)
        : sessions.data

      for (const session of targets) {
        try {
          const fileList = newFiles.map(f => `dictations/${f}`).join(", ")
          await client.session.prompt({
            path: { id: session.id },
            body: {
              noReply: true,
              parts: [{
                type: "text",
                text: `[DICTATION NOTIFICATION] New dictation file(s): ${fileList} — Max has written new instructions. Read immediately before continuing other work.`,
              }],
            },
          })
        } catch {
          // Session might not be active — that's fine
        }
      }
    } finally {
      notifying = false
    }
  }

  return {
    // Poll dictations/ after each tool call — reliable on all platforms
    "tool.execute.after": async (input) => {
      await checkForNewDictations(input.sessionID)
    },
  }
}
