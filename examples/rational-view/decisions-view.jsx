// @rational-view {"packages":{},"styles":[]}
import { useEffect, useMemo, useRef, useState } from 'react';

// Rational's decision API: System One format, routed to the registered decision model.
const DECISION_API = '/api/compat/v0/typesafe/v1/systemone';
const MODEL = 'rmdm-gemma4-12b';

// Question sets per photo category. `examples: true` turns every resource of the "Examples" category into an option
// defined by its picture.
const QUESTION_SETS = {
  Labels: [
    { key: 'cabinet', type: 'choice', title: 'Which cabinet does it go in?',
      instructions: 'Company rule: any product with a toxic (skull and crossbones) pictogram goes to the toxics cabinet, even if it is also flammable. Other flammable products go to the flammables cabinet, everything else to the general shelf. Where does the product in `photo` go?',
      criteria: { 'toxics cabinet': 'Toxics cabinet.', 'flammables cabinet': 'Flammables cabinet.', 'general shelf': 'General shelf.' } },
    { key: 'flame', type: 'noul', title: 'Is there a flame pictogram?',
      instructions: 'Does the label in `photo` show the flame pictogram?' },
    { key: 'hazard', type: 'score', title: 'How hazardous is it to handle?',
      instructions: 'How hazardous is the product in `photo` to handle?', criteria: ['Low', 'Medium', 'High'] },
  ],
  Signs: [
    { key: 'sign', type: 'choice', title: 'Which sign is visible?', examples: true,
      instructions: 'Which road sign is visible in `photo`?', criteria: { none: 'No road sign is visible.' } },
    { key: 'visibility', type: 'score', title: 'How visible is it?',
      instructions: 'How clearly visible is the road sign in `photo`?', criteria: ['Absent', 'Hard to read', 'Clear'] },
    { key: 'position', type: 'choice', title: 'Where is it?',
      instructions: 'Where is the road sign in `photo`?',
      criteria: { left: 'Left third of the image.', middle: 'Middle third of the image.', right: 'Right third of the image.', none: 'There is no road sign.' } },
  ],
};

const css = `
.rv { font: 14px/1.45 Inter, -apple-system, "Segoe UI", sans-serif; color: #050a0a; padding: 16px; display: grid; grid-template-columns: 220px minmax(0, 1fr) minmax(0, 1.1fr); gap: 16px; min-height: 100%; box-sizing: border-box; background: #f5f8f7; }
.rv * { box-sizing: border-box; }
.rv h3 { font-size: 11px; letter-spacing: .07em; text-transform: uppercase; color: #5d6363; margin: 0 0 8px; font-weight: 600; }
.rv .col { min-width: 0; }
.rv .group { margin-bottom: 14px; }
.rv .thumbs { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.rv .thumb { border: 2px solid transparent; border-radius: 8px; overflow: hidden; cursor: pointer; background: #fff; padding: 0; height: 92px; display: flex; align-items: center; justify-content: center; }
.rv .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.rv .thumb.on { border-color: #00a398; }
.rv .card { background: #fff; border: 1px solid #e1e7e6; border-radius: 12px; padding: 14px; }
.rv .photo { display: flex; justify-content: center; background: #eef5f3; border-radius: 10px; overflow: hidden; }
.rv .photo img { max-width: 100%; max-height: 62vh; display: block; }
.rv .name { margin: 10px 0 0; font-weight: 600; }
.rv .muted { color: #5d6363; font-size: 13px; }
.rv button.go { background: #00a398; color: #fff; border: 0; border-radius: 8px; padding: 10px 18px; font: inherit; font-weight: 600; cursor: pointer; }
.rv button.go:disabled { opacity: .5; cursor: default; }
.rv .q { border-top: 1px solid #e1e7e6; padding-top: 12px; margin-top: 12px; }
.rv .q:first-of-type { border-top: 0; margin-top: 0; padding-top: 0; }
.rv .qt { font-weight: 600; margin-bottom: 6px; display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }
.rv .chip { background: #11615c; color: #fff; border-radius: 6px; padding: 2px 9px; font-size: 13px; white-space: nowrap; }
.rv .bar { display: grid; grid-template-columns: minmax(0, 180px) minmax(0, 1fr) 40px; gap: 8px; align-items: center; margin: 5px 0; font-size: 13px; }
.rv .bar .lbl { display: flex; align-items: center; gap: 6px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.rv .bar .lbl img { width: 24px; height: 24px; object-fit: contain; }
.rv .track { height: 10px; background: #e6eeec; border-radius: 5px; overflow: hidden; }
.rv .fill { height: 100%; background: #00a398; }
.rv .num { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
.rv .foot { margin-top: 14px; display: flex; gap: 14px; flex-wrap: wrap; color: #5d6363; font-size: 12px; }
.rv .err { color: #b3261e; }
@media (max-width: 900px) { .rv { grid-template-columns: minmax(0, 1fr); } }
`;

const fmt = (p) => (typeof p === 'number' ? p.toFixed(2) : '');
const label = (r) => String(r?.name || '').replace(/\.[a-z0-9]+$/i, '').replace(/[-_]+/g, ' ');
const isImage = (r) => r && r.file && /^image\//.test(r.file.mimeType || '');

async function download(fileId) {
  const res = await fetch(`/api/knowledge/v0/files/${encodeURIComponent(fileId)}/download`, { credentials: 'include' });
  if (!res.ok) throw new Error(`image not available (${res.status})`);
  return res.blob();
}
const toAsset = (blob) => new Promise((ok, ko) => {
  const r = new FileReader();
  r.onload = () => ok({ type: 'image', format: /png/.test(blob.type) ? 'png' : 'jpg', data_base64: String(r.result).split(',')[1] });
  r.onerror = () => ko(new Error('could not read the image'));
  r.readAsDataURL(blob);
});

function View({ data }) {
  const resources = (data?.resourcesPage?.items || data?.resources || []).filter(isImage);
  const [urls, setUrls] = useState({});
  const blobs = useRef({});
  const [selected, setSelected] = useState(null);
  const [state, setState] = useState({ busy: false, error: null, result: null, ms: 0 });

  useEffect(() => {
    let alive = true; const made = [];
    resources.forEach((r) => {
      download(r.file.id).then((b) => {
        if (!alive) return; blobs.current[r.id] = b; const u = URL.createObjectURL(b); made.push(u);
        setUrls((m) => ({ ...m, [r.id]: u }));
      }).catch(() => {});
    });
    return () => { alive = false; made.forEach((u) => URL.revokeObjectURL(u)); };
  }, [resources.map((r) => r.id).join(',')]);

  const groups = useMemo(() => {
    const g = {}; resources.forEach((r) => { (g[r.category || 'Other'] ||= []).push(r); }); return g;
  }, [resources]);
  const examples = groups['Examples'] || [];
  const photo = resources.find((r) => r.id === selected) || (groups['Labels'] || groups['Signs'] || [])[0];
  const questions = QUESTION_SETS[photo?.category] || [];

  async function decide() {
    if (!photo) return;
    setState({ busy: true, error: null, result: null, ms: 0 });
    const started = performance.now();
    try {
      const assets = { PHOTO: await toAsset(blobs.current[photo.id] || await download(photo.file.id)) };
      const body = { model: MODEL, state: { photo: 'PHOTO' }, assets, questions: {} };
      for (const q of questions) {
        let criteria = q.criteria;
        if (q.examples) {
          // each example picture becomes an option, defined only by its image
          const pics = {};
          for (const ex of examples) {
            const ref = `EXAMPLE-${ex.id}`;
            assets[ref] = await toAsset(blobs.current[ex.id] || await download(ex.file.id));
            pics[label(ex)] = ref;
          }
          criteria = { ...pics, ...q.criteria };
        }
        body.questions[q.key] = { type: q.type, instructions: q.instructions, ...(criteria ? { criteria } : {}) };
      }
      const res = await fetch(DECISION_API, { method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json?.detail?.[0]?.msg || json?.error || `error ${res.status}`);
      setState({ busy: false, error: null, result: json, ms: performance.now() - started });
    } catch (e) {
      setState({ busy: false, error: e.message, result: null, ms: 0 });
    }
  }

  const exampleUrl = (name) => { const ex = examples.find((e) => label(e) === name); return ex ? urls[ex.id] : null; };
  const renderAnswer = (q) => {
    const a = state.result?.answers?.[q.key]; if (!a) return null;
    let rows, chip;
    if (a.type === 'noul') { rows = [['yes', a.noul], ['no', 1 - a.noul]]; chip = a.noul >= 0.5 ? `yes · ${fmt(a.noul)}` : `no · ${fmt(1 - a.noul)}`; }
    else if (a.type === 'score') { rows = Object.entries(a.probabilities).map(([k, p]) => [q.criteria[+k], p]); chip = `${q.criteria[a.score]} · ${fmt(a.probabilities[String(a.score)])}`; }
    else { rows = Object.entries(a.probabilities); chip = `${a.choice} · ${fmt(a.probabilities[a.choice])}`; }
    return (
      <div className="q" key={q.key}>
        <div className="qt"><span>{q.title}</span><span className="chip">{chip}</span></div>
        {rows.map(([name, p]) => (
          <div className="bar" key={name}>
            <span className="lbl">{exampleUrl(name) ? <img src={exampleUrl(name)} alt="" /> : null}{name}</span>
            <div className="track"><div className="fill" style={{ width: `${(p * 100).toFixed(1)}%` }} /></div>
            <span className="num">{fmt(p)}</span>
          </div>
        ))}
      </div>
    );
  };

  if (!data) return <div className="rv"><style>{css}</style><p className="muted">Loading…</p></div>;
  return (
    <div className="rv">
      <style>{css}</style>
      <div className="col">
        {['Labels', 'Signs', 'Examples'].filter((c) => groups[c]).map((c) => (
          <div className="group" key={c}>
            <h3>{c === 'Examples' ? 'Examples for the options' : c}</h3>
            <div className="thumbs">
              {groups[c].map((r) => (
                <button key={r.id} className={`thumb${photo?.id === r.id ? ' on' : ''}`} disabled={c === 'Examples'}
                  onClick={() => { setSelected(r.id); setState({ busy: false, error: null, result: null, ms: 0 }); }} title={label(r)}>
                  {urls[r.id] ? <img src={urls[r.id]} alt={label(r)} style={c === 'Examples' ? { objectFit: 'contain' } : null} /> : <span className="muted">…</span>}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="col card">
        <h3>Photo</h3>
        <div className="photo">{photo && urls[photo.id] ? <img src={urls[photo.id]} alt={label(photo)} /> : <p className="muted">Loading photo…</p>}</div>
        <p className="name">{photo ? label(photo) : 'No photo'}</p>
        <p className="muted">{photo?.notes || ''}</p>
        <button className="go" onClick={decide} disabled={!photo || state.busy}>{state.busy ? 'The model is reading the photo…' : `Decide (${questions.length} questions)`}</button>
      </div>
      <div className="col card">
        <h3>Decisions · {MODEL}</h3>
        {state.error ? <p className="err">{state.error}</p> : null}
        {!state.result && !state.error ? <p className="muted">Answers appear here: the chosen option and the probability of each option. No text is generated.</p> : null}
        {state.result ? questions.map(renderAnswer) : null}
        {state.result ? (
          <div className="foot">
            <span>{(state.ms / 1000).toFixed(1)} s</span>
            <span>{state.result.usage?.input_tokens} input tokens</span>
            <span>{state.result.usage?.output_tokens} positions read, 0 words generated</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}
