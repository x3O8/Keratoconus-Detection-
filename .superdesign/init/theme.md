# Theme

## Compact token summary

- Background: `#F4F7F6`
- Surface: `#FFFFFF`
- Primary ink: `#102521`
- Muted text: `#6D7D79`
- Border: `#DFE7E4`
- Primary green: `#0B6B57`
- Sidebar green: `#0F2824`
- Mint tint: `#D9F0E9`
- Warning surface: `#FFF8F3`; warning border `#F0D9CA`; warning text `#85533D`
- Typeface: current browser shell uses Arial; document loads DM Sans and Manrope. New React target should use Manrope for headings and DM Sans for body/UI.
- Radius: 8px small controls; 12px inputs/notices; 16px panels; 18px result hero; pill radius 999px.
- Shadow: current UI is border-led and largely shadowless. New UI may use a restrained `0 10px 30px rgba(16,37,33,.06)` only for elevated result/overlay surfaces.
- Spacing: 4px base; common gaps 8, 12, 16, 18, 24, 32, 48px.
- Breakpoints: 900px collapses shell; 520px stacks upload and result content.

## Raw CSS source

Source: `web/styles.css`

```css
:root{--bg:#f4f7f6;--surface:#fff;--ink:#102521;--muted:#6d7d79;--line:#dfe7e4;--green:#0b6b57;--mint:#d9f0e9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,sans-serif}
.shell{display:grid;grid-template-columns:238px 1fr;min-height:100vh}
aside{position:fixed;width:238px;height:100vh;background:#0f2824;color:white;padding:30px 20px;display:flex;flex-direction:column}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:24px;margin-bottom:18px}
.primary{width:100%;border:0;border-radius:12px;background:var(--green);color:#fff;padding:16px;font-weight:600}
.research{padding:8px 12px;border:1px solid #e8cbbd;color:#a84b28;border-radius:99px;font-size:11px;font-weight:700}
@media(max-width:900px){.shell{display:block}aside{position:static;width:100%;height:auto}.result-grid,.compact{grid-template-columns:1fr}}
@media(max-width:520px){.drop-grid,.feature-grid{grid-template-columns:1fr}.result-hero{grid-template-columns:1fr;text-align:center}}
```
