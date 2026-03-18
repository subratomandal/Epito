// --- Answer Engine
// Extracts key terms from a query, finds occurrences in note chunks,
// and returns focused 200-word excerpts around each match as LLM context.
// Less noise than full chunks = better model accuracy.

const STOP_WORDS = new Set([
  'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'which', 'who',
  'where', 'when', 'how', 'do', 'does', 'did', 'in', 'on', 'at', 'to',
  'for', 'of', 'and', 'or', 'this', 'that', 'all', 'any', 'my', 'your',
  'me', 'it', 'they', 'be', 'been', 'have', 'has', 'had', 'not', 'can',
  'could', 'will', 'would', 'with', 'from', 'by', 'about', 'into',
  'give', 'tell', 'list', 'listed', 'mentioned', 'named', 'written',
  'here', 'there', 'some', 'many', 'much', 'more', 'most', 'other',
  'than', 'then', 'also', 'just', 'but', 'if', 'so', 'no', 'yes',
  'i', 'you', 'we', 'he', 'she', 'its', 'them', 'their', 'our',
  'been', 'being', 'may', 'might', 'must', 'shall', 'should', 'need',
  'up', 'out', 'off', 'over', 'only', 'very', 'such', 'like',
  'what', 'name', 'names', 'please', 'paper', 'note', 'notes',
]);

// --- Term Extraction

// Simple suffix stemmer: "universities" -> "university", "mentioned" -> "mention"
function simpleStem(word: string): string {
  if (word.endsWith('ies') && word.length > 4) return word.slice(0, -3) + 'y';
  if (word.endsWith('es') && word.length > 4) return word.slice(0, -2);
  if (word.endsWith('ed') && word.length > 4) return word.slice(0, -2);
  if (word.endsWith('ing') && word.length > 5) return word.slice(0, -3);
  if (word.endsWith('tion') && word.length > 5) return word.slice(0, -4) + 'te';
  if (word.endsWith('s') && !word.endsWith('ss') && word.length > 3) return word.slice(0, -1);
  return word;
}

function extractQueryTerms(query: string): string[] {
  const words = query.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2 && !STOP_WORDS.has(w));

  const withStems: string[] = [];
  for (const w of words) {
    withStems.push(w);
    const stemmed = simpleStem(w);
    if (stemmed !== w && stemmed.length > 2) withStems.push(stemmed);
  }

  // Include 2-3 word ngrams for multi-word term matching
  const cleaned = query.toLowerCase().replace(/[^\w\s]/g, ' ');
  const allWords = cleaned.split(/\s+/).filter(w => w.length > 0);
  const phrases: string[] = [];

  for (let i = 0; i < allWords.length - 1; i++) {
    const bigram = `${allWords[i]} ${allWords[i + 1]}`;
    if (!STOP_WORDS.has(allWords[i]) || !STOP_WORDS.has(allWords[i + 1])) {
      phrases.push(bigram);
    }
    if (i < allWords.length - 2) {
      const trigram = `${allWords[i]} ${allWords[i + 1]} ${allWords[i + 2]}`;
      phrases.push(trigram);
    }
  }

  // Longest terms first so phrase matches take priority over single words
  const all = [...new Set([...phrases, ...withStems])];
  all.sort((a, b) => b.length - a.length);
  return all;
}

// --- Match Extraction

interface ContextMatch {
  term: string;
  position: number;
  excerpt: string;
}

function findMatchesWithContext(
  text: string,
  terms: string[],
  windowWords: number = 100,
): ContextMatch[] {
  const textLower = text.toLowerCase();
  const words = text.split(/\s+/);
  const matches: ContextMatch[] = [];
  const coveredRanges: [number, number][] = [];

  for (const term of terms) {
    let searchFrom = 0;

    while (true) {
      const pos = textLower.indexOf(term, searchFrom);
      if (pos === -1) break;
      searchFrom = pos + term.length;

      const charsBefore = text.slice(0, pos);
      const wordIdx = charsBefore.split(/\s+/).length - 1;

      // Skip if this region overlaps with an already-extracted excerpt
      const start = Math.max(0, wordIdx - windowWords);
      const end = Math.min(words.length, wordIdx + windowWords);

      const overlaps = coveredRanges.some(
        ([cs, ce]) => start < ce && end > cs
      );

      if (!overlaps) {
        const excerpt = words.slice(start, end).join(' ');
        matches.push({ term, position: pos, excerpt });
        coveredRanges.push([start, end]);
      }
    }
  }

  matches.sort((a, b) => a.position - b.position);
  return matches;
}

function countOccurrences(text: string, term: string): number {
  const lower = text.toLowerCase();
  const t = term.toLowerCase();
  let count = 0;
  let pos = 0;
  while ((pos = lower.indexOf(t, pos)) !== -1) {
    count++;
    pos += t.length;
  }
  return count;
}

// --- Context Extraction

export interface ExtractionResult {
  focusedContext: string;
  matchCount: number;
  matchedTerms: string[];
  directAnswer: string | null;
}

export function extractFocusedContext(
  query: string,
  chunks: string[],
): ExtractionResult {
  if (!chunks || chunks.length === 0) {
    return { focusedContext: '', matchCount: 0, matchedTerms: [], directAnswer: null };
  }

  const combined = chunks.join('\n\n');
  const terms = extractQueryTerms(query);

  if (terms.length === 0) {
    // No meaningful terms; fall back to first 500 words
    const words = combined.split(/\s+/);
    return {
      focusedContext: words.slice(0, 500).join(' '),
      matchCount: 0,
      matchedTerms: [],
      directAnswer: null,
    };
  }

  const foundTerms: string[] = [];
  const termCounts = new Map<string, number>();

  for (const term of terms) {
    const count = countOccurrences(combined, term);
    if (count > 0) {
      // Skip single words already covered by a longer phrase match
      const alreadyCovered = foundTerms.some(
        ft => ft.length > term.length && ft.includes(term)
      );
      if (!alreadyCovered) {
        foundTerms.push(term);
        termCounts.set(term, count);
      }
    }
  }

  if (foundTerms.length === 0) {
    return { focusedContext: '', matchCount: 0, matchedTerms: [], directAnswer: null };
  }

  const matches = findMatchesWithContext(combined, foundTerms, 100);
  const totalMatches = foundTerms.reduce((sum, t) => sum + (termCounts.get(t) || 0), 0);

  let focusedContext: string;
  if (matches.length === 0) {
    // Text too short for windowed extraction; use it all
    focusedContext = combined;
  } else {
    focusedContext = matches.map((m, i) => `[${i + 1}] ${m.excerpt}`).join('\n\n');
  }

  // All queries go through the AI model; this only provides focused context
  const directAnswer: string | null = null;

  console.log(
    `[AnswerEngine] Query: "${query}" | Terms: [${foundTerms.slice(0, 5).join(', ')}] | ` +
    `Matches: ${totalMatches} | Excerpts: ${matches.length} | ` +
    `Context: ${focusedContext.split(/\s+/).length} words`
  );

  return {
    focusedContext,
    matchCount: totalMatches,
    matchedTerms: foundTerms,
    directAnswer,
  };
}
