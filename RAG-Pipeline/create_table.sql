-- pgvector 확장
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table if missing
CREATE TABLE IF NOT EXISTS public.documents (
  id text PRIMARY KEY,
  content text,
  metadata jsonb DEFAULT '{}'::jsonb,
  embedding vector(4096),  -- 모델 차원에 맞게 조정
  created_at timestamptz DEFAULT now()
);

-- 필요한 컬럼만 추가
ALTER TABLE public.documents
  ADD COLUMN IF NOT EXISTS content text,
  ADD COLUMN IF NOT EXISTS metadata jsonb DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now(),
  ADD COLUMN IF NOT EXISTS embedding_1024 vector(1024),  -- 차원은 모델에 맞게 (일단 solar-passage 모델 사용)
  DROP COLUMN IF EXISTS embedding;

-- 임베딩 검색 최적화
CREATE INDEX IF NOT EXISTS idx_documents_embedding
ON public.documents USING ivfflat (embedding_1024 vector_l2_ops) WITH (lists = 100);