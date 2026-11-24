# PCI Agent

**Public edition · GitHub publisher [alex-yimingyang](https://github.com/alex-yimingyang)**

PCI Agent assesses the Pavement Condition Index (PCI) of an asphalt road sample unit using your own images or masks. OpenAI Codex and Anthropic Claude connect to the same local Python tools through the Model Context Protocol (MCP). Choose a method according to the evidence you can supply.

| Method | Input | Output |
|---|---|---|
| M1 · Direct visual estimate | Original RGB image | Whole-unit visual PCI interval with evidence for each endpoint |
| M2 · Detection from RGB | Original RGB image | Model-drawn distress geometry, per-instance severity, physical quantities and rule-based PCI |
| M3 · Unlabeled SAM3 masks | Binary prediction masks from your SAM3 workflow, optional aligned RGB for severity | Classes inferred from mask shape, measured distress quantities and rule-based PCI |
| M4 · Labeled masks | Instance masks with supplied classes, optional aligned RGB | Per-instance severity, measured quantities and rule-based PCI using the supplied classes |

The package includes code and the road deduct-curve configuration used for scoring. Supply your own pavement imagery, masks and scale. Geometry and scoring run locally. During a visual assessment, evidence returned to your signed-in chat can be processed by the model provider. The scoring tools require no separate model API key.

## 1. Install and check the tools

Use macOS or Linux with Python 3.12. On macOS with Homebrew, run `brew install python@3.12` if `python3.12` is unavailable. Extract the release ZIP and open its `PCI-Agent` folder. If the repository has been published under the name shown below, you can also obtain the same source with:

```sh
git clone https://github.com/alex-yimingyang/PCI-Agent.git
cd PCI-Agent
```

Run the following from the folder containing `launch-mcp.sh`. To keep inputs and run records local, store your road data outside the repository. Each new run writes its outputs under `runs/` by default.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-images.txt
export PCI_IMAGE_PYTHON="$PWD/.venv/bin/python"
```

Verify the four methods before signing into a model client. These tests create small synthetic geometries in temporary directories, exercise the MCP tools through a real stdio connection, and check the calculations. Both commands should finish with exit code 0. A separate native-window test is available for large images.

```sh
.venv/bin/python -B tests/verify_image_agents.py
.venv/bin/python -B tests/verify_image_detection.py
# Optional large-image and mask-alignment checks
.venv/bin/python -B tests/verify_native_windows.py
```

## 2A. Run with OpenAI Codex

**Terminal.** If Codex CLI is already available, check it with `codex --version`. To install it on macOS with Homebrew, run `brew install --cask codex`. On Linux with npm, use `npm install -g @openai/codex`. Run `codex` once, choose **Sign in with ChatGPT**, then exit to the terminal. [Codex CLI installation and sign-in](https://github.com/openai/codex#quickstart)

From the extracted package folder, start one method. Choose one line per session.

```sh
export PCI_IMAGE_PYTHON="$PWD/.venv/bin/python"
sh ./launch-chat.sh --agent M1 "Use M1. Ask for my original road image and physical scale. Respond in English."
sh ./launch-chat.sh --agent M2 "Use M2. Detect distress from original RGB. Ask for my inputs in English."
sh ./launch-chat.sh --agent M3 "Use M3. Classify unlabeled SAM3 masks first. Ask for my inputs in English."
sh ./launch-chat.sh --agent M4 "Use M4. Keep the supplied mask classes. Ask for my inputs in English."
```

Enter `/mcp` in the session and look for `road_pci`. For M3, the available tools should include `prepare_unlabeled_mask_agent` and `read_mask_grading_crops`. The launcher configures the service for this session. Each command opens a new chat dedicated to one method. [Codex MCP configuration](https://developers.openai.com/codex/mcp/)

**Codex desktop.** To make the service available in a desktop chat, open the extracted folder as a trusted project. Merge the following entry into its `.codex/config.toml`. Create the directory and file if needed. Replace both occurrences of `/absolute/path/to/PCI-Agent` with the actual folder printed by `pwd`. Start a new local chat and select the method by name in your request. [Codex project-scoped MCP configuration](https://developers.openai.com/codex/mcp/)

```toml
[mcp_servers.road_pci]
command = "/bin/sh"
args = ["/absolute/path/to/PCI-Agent/launch-mcp.sh"]
startup_timeout_sec = 30
tool_timeout_sec = 620

[mcp_servers.road_pci.env]
PCI_IMAGE_PYTHON = "/absolute/path/to/PCI-Agent/.venv/bin/python"
```

## 2B. Run with Claude

**Claude Code in Terminal.** Check `claude --version`. On macOS with Homebrew, install it with `brew install --cask claude-code` if needed. Linux users can follow Anthropic’s native installer instructions. Start `claude` once and follow the sign-in prompts, then exit to the terminal. [Claude Code installation and sign-in](https://code.claude.com/docs/en/quickstart)

From the extracted package folder, register the service once. Local scope makes the registration private to your account in this project. The quoted paths also work when the folder name contains spaces. [Claude Code local MCP servers](https://code.claude.com/docs/en/mcp)

```sh
claude mcp add --scope local --transport stdio road_pci -- /usr/bin/env \
  "PCI_IMAGE_PYTHON=$PWD/.venv/bin/python" /bin/sh "$PWD/launch-mcp.sh"
claude mcp get road_pci
claude
```

Look for **Connected** in the terminal status. In the Claude session, enter `/mcp` to inspect the tools. All four methods are available through this registration. Start a fresh chat for each independent assessment and use a prompt from section 3.

**Claude Desktop on macOS.** Install and open Claude Desktop, then go to **Claude → Settings → Developer → Edit Config**. Add the `road_pci` entry under `mcpServers` in `~/Library/Application Support/Claude/claude_desktop_config.json`. Merge it with any existing servers. Replace both absolute paths, save the file, fully quit Claude Desktop, and reopen it. In a new chat, check for the tools and give Claude an absolute path to your sample. [Claude Desktop local server setup](https://modelcontextprotocol.io/docs/develop/connect-local-servers)

```json
{
  "mcpServers": {
    "road_pci": {
      "command": "/bin/sh",
      "args": ["/absolute/path/to/PCI-Agent/launch-mcp.sh"],
      "env": {
        "PCI_IMAGE_PYTHON": "/absolute/path/to/PCI-Agent/.venv/bin/python"
      }
    }
  }
}
```

## 3. Assess your first new sample

Use the same sample boundary, original image resolution and physical scale for all four methods. An ROI is a binary mask identifying the pavement included in the density denominator. If the sample has shoulders, background or transparent borders, check the ROI before comparing scores. For an independent visual comparison, start M1 and M2 before showing the session any report, distress labels or results from the other methods.

Replace the bracketed fields below. Each prompt works in either client.

| Method | Prompt to paste |
|---|---|
| **M1 · Direct visual PCI** | “Use M1 on [original image]. The scale is [value] mm/pixel and the pavement ROI is [path]. Inspect the whole SU. Give the visual PCI interval and condition rating, with separate image evidence for each endpoint.” |
| **M2 · Detection from original RGB** | “Use M2 on [original image] with [scale] and [ROI]. Detect and classify each distress from RGB. Mark linear crack centerlines and affected-area polygons. Check each instance crop, grade severity, measure quantities and density, then compute the PCI interval and rating.” |
| **M3 · Unlabeled SAM3 masks** | “Use M3 with [SAM3 binary instance manifest], [ROI] and [scale]. Infer and freeze classes from mask shape. Then use [aligned original RGB] to assess each instance’s surface severity. Return the PCI interval, rating, low-confidence instances and audit path.” |
| **M4 · Labeled masks** | “Use M4 with [labeled mask manifest], [aligned original RGB], [ROI] and [scale]. Preserve the supplied classes and boundaries. Grade each instance and review suspected false positives. Show the retained and screened inventories and their PCI intervals and ratings.” |

**Physical scale.** Supply meters per native pixel or the measured area of the same pavement ROI. For 2 mm/pixel, the tool input is `{"status":"known","m_per_px":0.002}`. If scale is unavailable, say “Use pixel units.” The generic M2–M4 tools then report pixel quantities and a `pci_proxy_interval` normalized to a 5,000 ft² reference area. M1 gives a visual interval. For resized images, convert the scale to the pixel size of the image actually supplied.

**Mask format.** Use a single-channel binary image with zero background and one nonzero foreground value. One file per instance is easiest to audit. For a cropped mask, supply `bounds_xyxy` in original-image pixels, with exclusive right and bottom edges. M3 needs a full-canvas ROI when only cropped masks are supplied. A full-canvas mask can omit the bounds. Here is one M3 manifest entry:

```json
[{"path":"/data/SU21/mask_001.png","bounds_xyxy":[400,200,520,350],"semantics":"crack_trace"}]
```

For M4, add a `kind` field such as `"linear_crack"`. Other supported values are `alligator`, `block_crack`, `patch` and `pothole`. Use `affected_region` for a filled damaged area. Linear cracks use skeleton length, area distresses use affected surface area, and potholes use counts. Give each instance a supported grade or an adjacent severity range. Record assumptions and confidence when visible surface evidence is limited.

**M4 false-positive review.** Grade every candidate first and keep its ID, class, boundary, severity range and rationale. Review the candidates, then score the retained and screened inventories in separate runs when both scenarios are requested. Use the same grades, scale and pavement ROI for both. Record the reason for each excluded instance.

## 4. Read and troubleshoot the output

- **Find a run.** Ask for the `run_id` and the absolute path of the result JSON. By default, each run has its own subdirectory under `runs/`. Use the interval’s `_condition` field for the English road-condition label. An interval spanning categories shows both endpoint labels, such as *Fair to Good*. Instance severity (L/M/H) describes one distress, while the PCI condition describes the whole SU.
- **Check an unusually wide interval.** Inspect the per-instance classes, severity ranges, measured quantities and density. Check the search status and endpoint gap if the solver reached its budget. For large images, follow the native-window review prompts before requesting the whole-unit score.
- **If the service fails to connect**, run the two installation checks in section 1 first. Check that the configured launcher path exists and that `.venv/bin/python` can import the image libraries with `.venv/bin/python -c 'import cv2, rasterio, PIL'`. Update absolute paths after moving the package. Restart the desktop client after editing its configuration.
- **If the returned score is a pixel proxy**, provide the native pixel scale or the measured area and matching ROI, and start a new run. If an M3 class change is rejected during RGB review, the mask classification has already been frozen. Start a new mask-only classification run to revise it.

The rule-based methods use the numeric configuration in `engine/curves/faa_road_ac_tables.json`. It records the curve source and interpolation assumptions. Report these outputs as research estimates. The visible-surface assessment does not measure hidden distress or pothole depth.
