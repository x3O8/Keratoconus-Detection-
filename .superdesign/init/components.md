# Shared UI Components

The repository does not currently contain a React component library. The browser UI is a single static document (`web/index.html`) enhanced by `web/app.js`; the Streamlit UI uses page-level Python render functions. The reusable browser patterns below are therefore the current source of truth for the React redesign.

## Upload card pattern

Source: `web/index.html`

```html
<section class="panel">
  <div class="panel-head">
    <div class="step">1</div>
    <div><h3>Corneal imaging</h3><p>Add one or more available topography maps.</p></div>
    <strong id="image-count">0 / 7 added</strong>
  </div>
  <div id="drop-grid" class="drop-grid"></div>
</section>
```

## Clinical CSV uploader

Source: `web/index.html`

```html
<label class="csv-drop">
  <input id="csv" type="file" accept=".csv" hidden>
  <span class="file-icon">+</span>
  <span><b id="csv-name">Drop clinical CSV here</b><small id="csv-note">or click to browse · CSV up to 10 MB</small></span>
  <button type="button">Choose file</button>
</label>
```

## Risk result card

Source: `web/index.html`

```html
<div class="result-hero">
  <div><span class="result-label">ENSEMBLE FINDING</span><h2 id="diagnosis">—</h2><p id="severity">—</p></div>
  <div class="risk-ring" id="risk-ring"><div><b id="probability">0%</b><span>KC probability</span></div></div>
  <div class="confidence"><span>MODEL CONFIDENCE</span><b id="confidence">—</b><small>Weighted agreement</small></div>
</div>
```

## Model status block

Source: `web/index.html`

```html
<div class="status"><i></i><span><b id="device">Starting…</b><small>Compute engine</small></span></div>
<div class="model-list"><span>Image model <b id="image-status">—</b></span><span>Clinical model <b id="tabular-status">—</b></span></div>
```
