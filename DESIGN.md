---
version: alpha
name: Clinical model workspace
description: A quiet, unbranded interface for clinical and corneal imaging models.
colors:
  background: "#f5f7f5"
  surface: "#ffffff"
  ink: "#17352d"
  muted: "#5f746b"
  border: "#dfe7e1"
  primary: "#226b50"
  navigation: "#ffffff"
  mint: "#e8f0e5"
  teal: "#357e65"
  warning: "#c47a2c"
  danger: "#b23a3a"
typography:
  display:
    fontFamily: 'Manrope, sans-serif'
  body:
    fontFamily: 'DM Sans, sans-serif'
  data:
    fontFamily: 'Consolas, monospace'
rounded:
  DEFAULT: "9px"
  card: "16px"
spacing:
  page-max: "1080px"
  nav-height: "56px"
  page-padding: "38px"
  mobile-padding: "18px"
components:
  button:
    height: "42px"
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface}"
  card:
    rounded: "{rounded.card}"
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
  field:
    height: "39px"
    textColor: "{colors.ink}"
  caption:
    textColor: "{colors.muted}"
  divider:
    backgroundColor: "{colors.border}"
  navigation:
    backgroundColor: "{colors.navigation}"
  navigation-selected:
    textColor: "{colors.primary}"
  hero:
    backgroundColor: "{colors.mint}"
  chart:
    backgroundColor: "{colors.teal}"
  warning:
    textColor: "{colors.warning}"
  error:
    textColor: "{colors.danger}"
---

# Workspace design system

## Overview

The visual reference is a corneal topography instrument: quiet measurement frames and precise numbers. The audience is researchers presenting and exploring keratoconus models on a laptop or desktop. All four views are working surfaces. English is the supported interface language; no Japan-market requirements are established.

The user explicitly requests no product name, reduced content, fewer buttons, and a less congested interface. They subsequently asked to restore the earlier landing hero and Start assessment button with more spacious layout. Retain the “Small details. Deeper insight.” headline and illustrative corneal map as that landing view's single introduction. Keep the navigation slim and light, with text-only destinations and an emerald active underline. No product branding, repeated introductions, decorative badges, or duplicate calls to action. Model names remain factual identifiers. Preserve separate clinical and image evidence streams and research-use context.

Runtime ownership: `frontend/src/premium.css` is the canonical application theme, loaded after the existing workflow stylesheet. This document mirrors its named variables: primary → `--green`, navigation → `--nav-background`, background → `--bg`, ink → `--ink`, muted → `--muted`, border → `--line`, mint → `--mint`, teal → `--teal`. Existing warning and danger variables live in `styles.css`. Change runtime and documentation together; no token generation adapter exists.

## Colors

Emerald identifies the active navigation destination and primary actions. Sage softens the hero. White cards carry actual data. Amber identifies external-evaluation limits; red is reserved for errors and screening findings. Color always accompanies a textual status. The contour illustration uses muted green washes, not a patient heatmap. This release supports a light workspace with a white top navigation bar and system forced colors.

## Typography

Manrope handles titles and model names; DM Sans handles controls and prose. Consolas is restrained to recorded AUROC values and source paths. Fonts are bundled locally to avoid network dependencies. Measurements use tabular numerals. Body text is 13–14px, with 12px supporting notes. Interface copy describes model behavior without equating probability with diagnostic certainty.

## Layout

A sticky white top navigation bar contains four text-only destinations. Use a 56px desktop bar and a 52px phone bar, a two-pixel emerald active underline, and visible inset keyboard focus. A single navigation component serves all viewport sizes. Model availability belongs to the model cards; a compact 32px connection strip retains research context. The main surface is at most 1080px wide. No toolbar duplicates the active view or its actions. The landing hero pairs its original headline and one Start assessment action with an explicitly illustrative corneal map. Give the hero generous inset space and 56px separation from the concise model cards and recorded evaluations below. Export appears only with evaluations; Print appears only with results. Below 850px use sticky horizontal navigation; below 620px the hero and cards stack with generous vertical gaps. Clinical fields remain directly available; threshold and fusion controls sit in an Analysis settings disclosure. Explanations show the six strongest clinical contributions, with the rest, sources, attention, and clinical context expandable. Input warnings and model disagreement remain visible. Native disclosures stay keyboard accessible and print their contents. Upload previews reserve tile geometry; loaders preserve button width.

## Elevation & Depth

Use surface color and one-pixel borders for hierarchy. Reserve shadows for result summaries and ephemeral feedback. No glossy or floating card decoration. Evidence disclosures expand in document flow; notifications overlay without shifting content.

## Shapes

Cards use 13–16px radii and controls 7–9px radii. A circular probability display summarizes the actual result. Evaluation scope remains explicit. A thin emerald underline identifies the active workspace destination.

## Components

Shared owners are documented in `UX-CONTRACT.md`. Buttons have explicit hover, focus, pressed, disabled, and busy states. Primary actions are emerald; secondary controls have neutral borders. Native selects intentionally retain operating-system popup behavior. Uploads provide keyboard-accessible file inputs, picker and drop entry, file type/size checks, remove/replace actions, and object URL cleanup. Notifications use one polite live region; correction errors remain inline.

Lucide supplies outlined icons; icons accompany labels except named utility controls. Motion is limited to 180ms disclosure and feedback transitions and the busy spinner. Reduced motion disables nonessential movement. Global scrollbars use the thumb, track, hover, and active tokens in `premium.css`, with standards properties, WebKit fallbacks, and system colors under forced colors.

Clinical AUROC and AUPRC are imported directly from repository JSON reports. Imaging accuracy and F1 come from the connected service checkpoint metadata, show a dash when absent, and are labelled saved checkpoint validation. Internal and external datasets remain distinct; the EfficientNetV2-S report is explicitly separate from the EfficientNet-B0 runtime. Illustrations are never shown as patient results.

## Do's and Don'ts

- Keep every metric connected to its dataset, checkpoint, and evaluation scope.
- Preserve entered assessment data during in-app navigation.
- Keep evidence and interpretation limitations visible in research presentations.
- Do not turn research scores into claims of clinical validation.
- Do not add decorative motion, neon, or screen-local copies of theme tokens.
- Do not reintroduce a product name, duplicate navigation, or repeated assessment actions. Keep the restored landing hero as the only presentation-style introduction.
