// Runs the AI off the main thread so the page stays responsive while it thinks.
// Message in:  {id, moves, human, sims}   Message out: {id, state}, state may be {error}.
importScripts("engine.js");

const ready = (async () => {
  const [manifest, weights] = await Promise.all([
    fetch("weights.json").then(r => r.json()),
    fetch("weights.bin").then(r => r.arrayBuffer()),
  ]);
  try {
    const wasm = await fetch("net.wasm").then(r => r.arrayBuffer());
    return await WasmNet.create(wasm, weights);
  } catch (e) {
    // No WebAssembly SIMD (very old browser): the plain-JS net gives the same moves, slower.
    return new Net(manifest, weights);
  }
})();

onmessage = async ({ data }) => {
  try {
    const net = await ready;
    postMessage({ id: data.id, state: compute(net, data.moves, data.human, data.sims) });
  } catch (e) {
    postMessage({ id: data.id, state: { error: String(e && e.message || e) } });
  }
};
