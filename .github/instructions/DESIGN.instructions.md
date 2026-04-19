# DESIGN.md

AstroPlanner uses a **desktop-first, data-dense astronomy planning UI** with **dark cyberpunk surfaces, luminous accents, and instrument-panel structure**. AI agents should read this file before generating or modifying UI so new screens look native to the app instead of generic web dashboards.

## Design intent

- The product should feel like **astronomy software for an active observing session**, not a marketing site or a consumer mobile app.
- The UI is **information-rich and multi-paneled**: users compare targets, plots, metrics, sky views, weather, and AI help in parallel.
- The default visual mood is **dark mode + Violet Pulse theme** (`DEFAULT_DARK_MODE = True`, `DEFAULT_UI_THEME = "violet"`), but the structure must also work across the other built-in themes.
- Visual style should read as **precise, technical, and slightly neon**, not playful, soft, or overly minimal.

## Core look and feel

### Overall style

- Prefer **layered dark surfaces** over flat black.
- Use **thin glowing borders**, not heavy outlines.
- Corners are **rounded but not bubbly**:
  - major cards/panels: around **16 px**
  - tabs and inputs: around **10-12 px**
  - chips/badges: around **9 px**
- Keep contrast high enough for nighttime use, but avoid harsh white-on-black everywhere.
- Accent color should appear as **controlled neon energy**, not full-screen gradients.

### Theme character

The existing themes are:

- `graphite` / **Graphite Grid**
- `laguna` / **Neon Harbor**
- `terracotta` / **Ember Circuit**
- `violet` / **Violet Pulse**
- `rosewood` / **Rose Noir**

Across all themes:

- `accent_primary` drives focus, selected state, and most chart emphasis
- `accent_secondary` drives hover glow, selected highlights, and secondary emphasis
- semantic states are stable:
  - info = accent-primary family
  - success = green
  - warning = amber/yellow
  - error = pink/rose

Do not invent unrelated color systems when adding UI. Reuse the existing theme tokens from `astroplanner/theme.py`.

## Layout grammar

### Main application layout

The main dashboard is built from persistent desktop panels, not stacked mobile sections:

1. top session/filter strip
2. main visibility plot card
3. sky/preview block (radar + cutout/finder)
4. night details / metrics card
5. targets table and actions

Follow these rules:

- Keep **multiple tools visible at once**.
- Prefer **side-by-side panels** over deep drill-down flows.
- New views should feel like part of an **instrument console**.
- Avoid huge empty hero areas or marketing-style whitespace.
- Avoid hiding critical astronomy context behind extra clicks when it can stay visible.

### Cards and surfaces

Existing major cards use object names like:

- `PlotCard`
- `TableCard`
- `PolarCard`
- `InfoCard`
- `CutoutFrame`
- `TopControlsBar`
- `ActionsBar`
- `RootContainer`

Use the same card language for new UI:

- title/header area at top
- compact content beneath
- subtle border glow
- slightly translucent/differentiated surface from the background

### Dialogs

- Dialogs should feel like **dense control panels**, not oversized blank forms.
- Use `_fit_dialog_to_screen(...)` from `astroplanner/ui/common.py` so dialogs stay usable on smaller displays.
- Keep content aligned to a clear grid with compact spacing and obvious action rows.

## Typography

- Use the existing **display font style** for section titles, tabs, and prominent buttons. The app uses a Rajdhani-style display face through `theme_utils.py`.
- Use the regular UI sans stack for body text, form controls, and dense data.
- Titles should feel **technical and crisp**, with slight extra letter spacing.
- Use **sentence case** for labels and actions. Do not switch to all-caps UI chrome.
- Keep helper text short and practical.

Use existing title/hint conventions:

- `SectionTitle` = prominent section label
- `SectionHint` = smaller supporting guidance / muted metadata

## Component rules

### Buttons

Buttons are not flat system buttons. They use gradients, rounded corners, and hover glow.

Use only the existing button variants:

- `primary` = main action in the area
- `secondary` = important supporting action
- `ghost` = utility / low-emphasis action

Implementation hooks:

- `_set_button_variant(button, "primary" | "secondary" | "ghost")`
- `_set_button_icon_kind(button, "...")`

Button rules:

- use **one dominant primary action** per local cluster
- secondary actions can sit nearby
- ghost actions are for tools, reset, export, utility, or non-destructive support
- do not create loud rainbow button sets
- do not mix too many equal-priority buttons in one row

### Labels and status text

Status labels should use semantic tone instead of ad hoc styling.

Implementation hook:

- `_set_label_tone(label, "info" | "success" | "warning" | "error" | "muted")`

Use tone to communicate state quickly while preserving a consistent palette.

### Tables

Tables are a primary interaction model in AstroPlanner.

Rules:

- favor **dense, sortable, scan-friendly tables**
- use alternating row surfaces and clear selection states
- preserve room for important astronomy columns rather than oversized padding
- action cells should feel integrated, not like big web buttons jammed into rows

When adding new table UIs:

- prefer `TargetTableView` or follow its density/interaction model
- preserve keyboard friendliness
- preserve hover and selection clarity

### Tabs

Tabs should feel like part of the control surface:

- compact
- display-font labeling
- integrated with the card they control
- selected tab blends into active pane

Do not build browser-like giant tabs or pill-tab marketing components.

### Chips and badges

Weather and status chips are small, rounded, and semantic. They are used to summarize context quickly.

Relevant properties:

- `weather_chip="true"`
- `weather_chip_role="weather|context|clock|solar|lunar"`
- `weather_chip_series="temp|wind|cloud|humidity|pressure"`

Use chips for short, glanceable state only. Do not turn long descriptions into pill clouds.

### Toggles and custom controls

Custom controls should feel “instrument-like,” similar to `NeonToggleSwitch`:

- compact
- slightly futuristic
- animated, but quickly
- clear checked/unchecked contrast

Motion should be subtle and purposeful, not decorative.

### Loading states

Prefer existing skeleton/shimmer placeholders over blank loading gaps:

- `SkeletonShimmerWidget`

Loading states should preserve the shape of the final UI so the layout does not jump.

## Charts, plots, and preview surfaces

Plots are a defining part of the product.

Rules:

- charts should feel like **serious observing tools**
- use theme-driven plot colors from `theme.py`
- keep grids, guides, crosshairs, and overlays legible but restrained
- selected target state may use `accent_secondary`
- key astronomical reference lines (sun, moon, altitude limit, twilight bands) must stay visually distinct

Preview/cutout/finder areas should:

- feel embedded in a card, not like detached image dumps
- preserve overlays and FOV clarity
- support tool-column controls without cluttering the main image

## Feature-specific guidance

### AI Assistant

The AI assistant is a **tool window**, not a chat app clone.

Keep this structure:

1. left vertical tool/action column
2. central transcript area
3. bottom composer row
4. lightweight runtime/status badge

The assistant should feel integrated with the planning workflow:

- practical and compact
- readable during streaming
- visually aligned with the rest of the app
- never styled like a bubbly consumer messenger

### Suggest Targets

This dialog is a **candidate review workspace**:

- keep summary + filters + sortable table visible
- emphasize scanability and direct add-to-plan actions
- filters should stay compact and aligned

### Weather workspace

Weather is a **data board**, not a decorative forecast app.

- use chips, compact summaries, and charts
- keep source/status information visible
- show partial-failure states clearly without collapsing the whole view

## Spacing and density

- Default to **compact desktop spacing**
- Use tighter spacing inside toolbars, filter rows, and action clusters
- Use a little more breathing room around major card boundaries and section headers
- Avoid oversized padding inside dense operational panels

If forced to choose, prefer **clarity through grouping** over extra whitespace.

## Interaction behavior

- Hover effects may glow, but only subtly
- Keep animations short and functional
- Preserve keyboard and power-user workflows
- Do not require mouse-only interactions for core tasks
- Avoid disruptive full-panel refreshes when a localized update is possible

## Copy style for UI text

- concise
- practical
- astronomy-task oriented
- sentence case

Good:

- `Describe Object`
- `Quick Targets`
- `Warm Up LLM`
- `Cloud Analysis`

Avoid:

- chatty instructional paragraphs inside the main UI
- vague labels like `Continue`, `Manage`, `Explore` without context
- marketing-style microcopy

## Implementation hooks for agents

When generating actual code, reuse the existing UI system instead of inventing parallel styling.

### Reuse these helpers

- `astroplanner/theme.py`
  - `build_stylesheet(...)`
  - `resolve_theme_tokens(...)`
- `astroplanner/ui/theme_utils.py`
  - `_set_button_variant(...)`
  - `_set_button_icon_kind(...)`
  - `_set_label_tone(...)`
  - `_style_dialog_button_box(...)`
- `astroplanner/ui/common.py`
  - `_fit_dialog_to_screen(...)`
  - `TargetTableView`
  - `SkeletonShimmerWidget`

### Reuse these naming/property conventions

- object names:
  - `SectionTitle`
  - `SectionHint`
  - `RootContainer`
  - `TopControlsBar`
  - `ActionsBar`
  - `PlotCard`
  - `TableCard`
  - `PolarCard`
  - `InfoCard`
  - `CutoutFrame`
- widget properties:
  - `variant`
  - `tone`
  - `accented`
  - `weather_chip`
  - `weather_chip_role`
  - `weather_chip_series`

### Localization

Visible UI text should remain compatible with the existing localization system. Reuse the app’s translation helpers rather than hard-coding styling-specific text behavior.

## Anti-patterns

Do **not** generate UI that looks like:

- a marketing landing page
- a mobile-first SaaS dashboard
- a soft pastel consumer app
- a plain native gray form with no theme integration
- a generic chat interface for the AI window

Specifically avoid:

- giant empty hero sections
- giant cards with one metric and lots of wasted space
- heavy drop shadows everywhere
- mismatched accent colors outside the theme system
- oversized rounded corners
- multi-step wizard flows for tasks that belong in one dense desktop panel

## One-sentence prompt summary for agents

When in doubt, generate **a compact, dark, desktop astronomy control surface with rounded neon-edged cards, dense tables, clear plots, Rajdhani-style headings, gradient action buttons, and theme-token-driven accents**.
