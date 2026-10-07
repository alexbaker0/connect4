// Connect Four AlphaZero net forward pass in C, compiled to WebAssembly SIMD.
// Same maths as Net.forward in engine.js (which is the fallback), ~4x faster.
//
// Build (clang from WinLibs LLVM):
//   clang --target=wasm32 -O3 -msimd128 -mbulk-memory -nostdlib -Wl,--no-entry
//         -o net.wasm src/net.c
//
// Usage from JS: copy weights.bin into weights() (300247 floats, BN folded,
// manifest order), call init(); then per position fill input() (2x42 planes)
// and call forward(); output() holds 7 logits + the pre-tanh value.

#include <wasm_simd128.h>

#define ROWS 6
#define COLS 7
#define NPX 42
#define CH 64
#define BLOCKS 4
#define PW 9
#define P 72                 /* padded 8x9 plane */
#define NW 300247

static float W[NW];
static float IN[2 * NPX];
static float OUT[8];

/* weights repacked with K padded to a multiple of 4 (only the stem needs it) */
static float stemw[CH * 20];
static float *stemb, *rw1[BLOCKS], *rb1[BLOCKS], *rw2[BLOCKS], *rb2[BLOCKS];
static float *pcw, *pcb, *pfw, *pfb, *vcw, *vcb, *v1w, *v1b, *v2w, *v2b;

static float A[CH * P], H[CH * P], X[2 * P];
static float COL[NPX * CH * 9];

#define EXPORT(n) __attribute__((export_name(n)))

EXPORT("weights") float *weights(void) { return W; }
EXPORT("input") float *input(void) { return IN; }
EXPORT("output") float *output(void) { return OUT; }

EXPORT("init") void init(void) {
  float *p = W;
  float *sw = p; p += CH * 2 * 9;
  stemb = p; p += CH;
  for (int i = 0; i < BLOCKS; i++) {
    rw1[i] = p; p += CH * CH * 9; rb1[i] = p; p += CH;
    rw2[i] = p; p += CH * CH * 9; rb2[i] = p; p += CH;
  }
  pcw = p; p += 2 * CH; pcb = p; p += 2;
  pfw = p; p += COLS * 2 * NPX; pfb = p; p += COLS;
  vcw = p; p += CH; vcb = p; p += 1;
  v1w = p; p += 64 * NPX; v1b = p; p += 64;
  v2w = p; p += 64; v2b = p; p += 1;
  for (int oc = 0; oc < CH; oc++)
    for (int j = 0; j < 20; j++) stemw[oc * 20 + j] = j < 18 ? sw[oc * 18 + j] : 0.0f;
}

static inline float hsum(v128_t v) {
  return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) +
         wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

/* out = relu(conv3x3(in) + b [+ in-place residual]); K = cin*9 rounded up to 4 */
static void conv3(const float *in, int cin, const float *w, const float *b, int K,
                  float *out, int residual) {
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++) {
      float *d = COL + (r * COLS + c) * K;
      int j = 0;
      for (int ic = 0; ic < cin; ic++) {
        const float *s = in + ic * P + r * PW + c;
        d[j++] = s[0];      d[j++] = s[1];          d[j++] = s[2];
        d[j++] = s[PW];     d[j++] = s[PW + 1];     d[j++] = s[PW + 2];
        d[j++] = s[2 * PW]; d[j++] = s[2 * PW + 1]; d[j++] = s[2 * PW + 2];
      }
      while (j < K) d[j++] = 0.0f;
    }
  static float tmp[CH * NPX];
  for (int oc = 0; oc < CH; oc += 4) {
    const float *w0 = w + oc * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
    for (int px = 0; px < NPX; px += 2) {
      const float *x0 = COL + px * K, *x1 = x0 + K;
      v128_t a00 = wasm_f32x4_splat(0), a01 = a00, a10 = a00, a11 = a00,
             a20 = a00, a21 = a00, a30 = a00, a31 = a00;
      for (int j = 0; j < K; j += 4) {
        v128_t u = wasm_v128_load(x0 + j), v = wasm_v128_load(x1 + j);
        v128_t p = wasm_v128_load(w0 + j), q = wasm_v128_load(w1 + j);
        v128_t s = wasm_v128_load(w2 + j), t = wasm_v128_load(w3 + j);
        a00 = wasm_f32x4_add(a00, wasm_f32x4_mul(p, u)); a01 = wasm_f32x4_add(a01, wasm_f32x4_mul(p, v));
        a10 = wasm_f32x4_add(a10, wasm_f32x4_mul(q, u)); a11 = wasm_f32x4_add(a11, wasm_f32x4_mul(q, v));
        a20 = wasm_f32x4_add(a20, wasm_f32x4_mul(s, u)); a21 = wasm_f32x4_add(a21, wasm_f32x4_mul(s, v));
        a30 = wasm_f32x4_add(a30, wasm_f32x4_mul(t, u)); a31 = wasm_f32x4_add(a31, wasm_f32x4_mul(t, v));
      }
      tmp[(oc + 0) * NPX + px] = hsum(a00); tmp[(oc + 0) * NPX + px + 1] = hsum(a01);
      tmp[(oc + 1) * NPX + px] = hsum(a10); tmp[(oc + 1) * NPX + px + 1] = hsum(a11);
      tmp[(oc + 2) * NPX + px] = hsum(a20); tmp[(oc + 2) * NPX + px + 1] = hsum(a21);
      tmp[(oc + 3) * NPX + px] = hsum(a30); tmp[(oc + 3) * NPX + px + 1] = hsum(a31);
    }
  }
  for (int oc = 0; oc < CH; oc++)
    for (int r = 0; r < ROWS; r++)
      for (int c = 0; c < COLS; c++) {
        int o = oc * P + (r + 1) * PW + c + 1;
        float v = tmp[oc * NPX + r * COLS + c] + b[oc];
        if (residual) v += out[o];
        out[o] = v > 0 ? v : 0;
      }
}

/* 1x1 conv + relu head, flattened channel-major like torch */
static void head(const float *w, const float *b, int nc, float *f) {
  for (int oc = 0; oc < nc; oc++)
    for (int r = 0; r < ROWS; r++)
      for (int c = 0; c < COLS; c++) {
        int pi = (r + 1) * PW + c + 1;
        float v = b[oc];
        for (int ic = 0; ic < CH; ic++) v += w[oc * CH + ic] * A[ic * P + pi];
        f[oc * NPX + r * COLS + c] = v > 0 ? v : 0;
      }
}

EXPORT("forward") void forward(void) {
  for (int i = 0; i < 2 * P; i++) X[i] = 0;
  for (int pl = 0; pl < 2; pl++)
    for (int r = 0; r < ROWS; r++)
      for (int c = 0; c < COLS; c++)
        X[pl * P + (r + 1) * PW + c + 1] = IN[pl * NPX + r * COLS + c];

  conv3(X, 2, stemw, stemb, 20, A, 0);
  for (int i = 0; i < BLOCKS; i++) {
    conv3(A, CH, rw1[i], rb1[i], CH * 9, H, 0);
    conv3(H, CH, rw2[i], rb2[i], CH * 9, A, 1);
  }

  float pf[2 * NPX], vf[NPX], h1[64];
  head(pcw, pcb, 2, pf);
  for (int o = 0; o < COLS; o++) {
    float v = pfb[o];
    for (int i = 0; i < 2 * NPX; i++) v += pfw[o * 2 * NPX + i] * pf[i];
    OUT[o] = v;
  }
  head(vcw, vcb, 1, vf);
  for (int o = 0; o < 64; o++) {
    float v = v1b[o];
    for (int i = 0; i < NPX; i++) v += v1w[o * NPX + i] * vf[i];
    h1[o] = v > 0 ? v : 0;
  }
  float v = v2b[0];
  for (int i = 0; i < 64; i++) v += v2w[i] * h1[i];
  OUT[7] = v;   /* tanh applied in JS */
}
