export interface Mode1Result {
  score: number;
  confidenceInterval: number;
  domain: string;
  verdict: string;
  modelGroup: string;
  keyMarkers: string[];
}

export interface CategoryAnalysis {
  category: string;
  score: string;
  weight: number;
  contribution: string;
}

export interface StructuralPattern {
  marker: string;
  quote: string;
  description: string;
}

export interface Mode2Result {
  score: number;
  confidenceInterval: number;
  domain: string;
  verdict: string;
  modelGroup: string;
  categories: CategoryAnalysis[];
  indicators: {
    ttr: number;
    lexicalDensity: number;
    hapaxLegomena: number;
    entropy: number;
    cvSentenceLength: number;
    dependencyDepth: number;
    hedgesCount: number;
    fleschEase: number;
    valenceRange: number;
  };
  structuralPatterns: StructuralPattern[];
  modelGroupReasoning: string;
  conclusion: string;
}

export interface Mode3Result {
  rewrittenText: string;
  changes: string[];
  scoreBefore: number;
  scoreAfter: number;
  confidenceIntervalAfter: number;
  verdictAfter: string;
}

export interface IterationResult {
  iteration: number;
  score: number;
  senseCheck: string;
  styleCheck: string;
  integrityCheck: string;
  aggressiveness: string;
  status: string;
  textSnapshot: string;
}

export interface Mode4Result {
  iterations: IterationResult[];
  finalText: string;
  scoreBefore: number;
  scoreAfter: number;
  totalIterations: number;
  qualityAssessment: string;
  stopReason: string;
}

export interface ComparisonItem {
  textIndex: number;
  title: string;
  score: number;
  confidenceInterval: number;
  verdict: string;
  domain: string;
  modelGroup: string;
  indicators: {
    ttr: number;
    lexicalDensity: number;
    entropy: number;
    cvSentenceLength: number;
    dependencyDepth: number;
    valenceRange: number;
  };
  structuralMarkers: string[];
}

export interface Mode5Result {
  comparisons: ComparisonItem[];
  conclusion: string;
}
