export class VectorIndex {
  private entries = new Map<string, Float32Array>();
  readonly dimensions: number;

  constructor(dimensions: number) {
    this.dimensions = dimensions;
  }

  get size(): number {
    return this.entries.size;
  }

  add(id: string, vector: number[]): void {
    this.entries.set(id, new Float32Array(l2Normalize(vector)));
  }

  addBatch(items: { id: string; vector: number[] }[]): void {
    for (const item of items) this.add(item.id, item.vector);
  }

  remove(id: string): void {
    this.entries.delete(id);
  }

  clear(): void {
    this.entries.clear();
  }

  search(query: number[], topK = 10, threshold = 0.0): { id: string; score: number }[] {
    const norm = new Float32Array(l2Normalize(query));
    const results: { id: string; score: number }[] = [];

    for (const [id, vec] of this.entries) {
      const score = dot(norm, vec);
      if (score >= threshold) results.push({ id, score });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < len; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom < 1e-10 ? 0 : d / denom;
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) sum += a[i] * b[i];
  return sum;
}

function l2Normalize(v: number[]): number[] {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm < 1e-10) return v;
  return v.map(x => x / norm);
}
