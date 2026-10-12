-- Allow each indexed document to carry the embedding model and its native dimensions.
ALTER TABLE chunks DROP CONSTRAINT chunks_embedding_model_check;
ALTER TABLE chunks ALTER COLUMN embedding TYPE vector;
CREATE INDEX chunks_embedding_model_idx ON chunks(embedding_model);
