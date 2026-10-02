# Shared Layouts

## Browser application shell

Source: `web/index.html`

```html
<div class="shell">
  <aside>
    <a class="brand" href="#"><span class="brandmark">K</span><span>KeratoScan<small>AI diagnostics</small></span></a>
    <nav>
      <button class="nav active" data-view="workspace"><span>New analysis</span></button>
      <button class="nav" data-view="results"><span>Results</span></button>
    </nav>
    <div class="side-bottom">
      <div class="status"><i></i><span><b id="device">Starting…</b><small>Compute engine</small></span></div>
      <div class="model-list"><span>Image model <b id="image-status">—</b></span><span>Clinical model <b id="tabular-status">—</b></span></div>
    </div>
  </aside>
  <main>
    <header>
      <div><p class="eyebrow">CLINICAL DECISION SUPPORT</p><h1>Corneal assessment</h1></div>
      <div class="research">Research use only</div>
    </header>
    <!-- active view content -->
  </main>
</div>
```

Desktop uses a fixed 238px dark-green sidebar and a fluid content column. At widths below 900px the sidebar becomes a compact horizontal bar and content becomes single-column.
