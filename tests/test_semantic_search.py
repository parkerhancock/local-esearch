"""Tests for semantic and hybrid search with embeddings."""

import sqlite3

import pytest
from local_esearch import Elasticsearch, TableIndex


class TestSemanticSearchWithVec:
    """Test semantic search modes with sqlite-vec."""

    def test_hybrid_search(self, es_with_embeddings):
        """Test hybrid search combines FTS5 and vector results."""
        es = es_with_embeddings

        # Index some documents
        es.index(index="test", id="1", document={"title": "Python", "body": "Learn Python"})
        es.index(index="test", id="2", document={"title": "ML", "body": "Neural networks"})
        es.index(index="test", id="3", document={"title": "Data", "body": "Python for data"})
        es.indices.refresh("test")

        # Hybrid search
        response = es.search(index="test", q="python", mode="hybrid")

        assert response["hits"]["total"]["value"] >= 1
        assert "hits" in response["hits"]

    def test_semantic_search(self, es_with_embeddings):
        """Test semantic-only search."""
        es = es_with_embeddings

        es.index(index="test", id="1", document={"title": "Python", "body": "Learn Python"})
        es.index(index="test", id="2", document={"title": "JavaScript", "body": "Modern JS"})
        es.indices.refresh("test")

        # Semantic search
        response = es.search(index="test", q="coding language", mode="semantic")

        assert "hits" in response
        # May or may not have results depending on embedding similarity

    def test_keyword_search_default(self, es_with_embeddings):
        """Test keyword search is still the default."""
        es = es_with_embeddings

        es.index(index="test", id="1", document={"title": "Python", "body": "Language"})
        es.indices.refresh("test")

        response = es.search(index="test", q="python")

        assert response["hits"]["total"]["value"] == 1


class TestTableIndexSemanticSearch:
    """Test semantic search with registered table indexes."""

    def test_table_index_hybrid_search(self, tmp_path):
        """Test hybrid search on a registered table."""
        db_path = tmp_path / "test.db"

        # Create table with data
        conn = sqlite3.connect(str(db_path))
        try:
            conn.enable_load_extension(True)
            import sqlite_vec

            sqlite_vec.load(conn)
        except Exception:
            conn.close()
            pytest.skip("sqlite-vec not available")

        conn.execute("""
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT
            )
        """)
        conn.execute("INSERT INTO articles VALUES (1, 'Python Guide', 'Learn Python basics')")
        conn.execute("INSERT INTO articles VALUES (2, 'ML Tutorial', 'Neural networks')")
        conn.execute("INSERT INTO articles VALUES (3, 'Data Analysis', 'Data science')")
        conn.commit()
        conn.close()

        # Create mock backend
        class MockBackend:
            backend_name = "mock"
            dimensions = 8
            model_name = "mock"

            def embed(self, text):
                h = hash(text) % 1000
                return [(h + i) / 1000.0 for i in range(8)]

            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]

        # Connect with ES and register table
        es = Elasticsearch(path=str(db_path), embedding_backend=MockBackend())
        es.register_table(
            index="articles",
            table="articles",
            id_column="id",
            text_columns=["title", "content"],
        )
        es.indices.reindex("articles")

        # Test hybrid search
        response = es.search(index="articles", q="python", mode="hybrid")

        assert response["hits"]["total"]["value"] >= 1

        # Check for inner_hits in results (from chunk matching)
        for hit in response["hits"]["hits"]:
            if "inner_hits" in hit:
                assert "chunks" in hit["inner_hits"]

        es.close()

    def test_table_index_semantic_only(self, tmp_path):
        """Test semantic-only search on a registered table."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(str(db_path))
        try:
            conn.enable_load_extension(True)
            import sqlite_vec

            sqlite_vec.load(conn)
        except Exception:
            conn.close()
            pytest.skip("sqlite-vec not available")

        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, text TEXT)")
        conn.execute("INSERT INTO docs VALUES (1, 'Hello world')")
        conn.execute("INSERT INTO docs VALUES (2, 'Goodbye world')")
        conn.commit()
        conn.close()

        class MockBackend:
            backend_name = "mock"
            dimensions = 8
            model_name = "mock"

            def embed(self, text):
                # Make "hello" and "greeting" have similar embeddings
                if "hello" in text.lower() or "greeting" in text.lower():
                    return [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                return [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]

        es = Elasticsearch(path=str(db_path), embedding_backend=MockBackend())
        es.register_table(index="docs", table="docs", text_columns=["text"])
        es.indices.reindex("docs")

        # Semantic search should find "Hello world" as closest to "greeting"
        response = es.search(index="docs", q="greeting", mode="semantic")

        assert "hits" in response
        es.close()


class TestTableIndexDirectAPI:
    """Test TableIndex class directly with embeddings."""

    def test_table_index_vector_search(self, tmp_path):
        """Test vector search on TableIndex directly."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(str(db_path))
        try:
            conn.enable_load_extension(True)
            import sqlite_vec

            sqlite_vec.load(conn)
        except Exception:
            conn.close()
            pytest.skip("sqlite-vec not available")

        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO docs VALUES (1, 'Python programming language')")
        conn.execute("INSERT INTO docs VALUES (2, 'JavaScript web development')")
        conn.commit()

        class MockBackend:
            backend_name = "mock"
            dimensions = 8
            model_name = "mock"

            def embed(self, text):
                h = hash(text) % 1000
                return [(h + i) / 1000.0 for i in range(8)]

            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]

        index = TableIndex(
            conn=conn,
            table="docs",
            id_column="id",
            text_columns=["content"],
            embedding_backend=MockBackend(),
            chunk_size=20,
            chunk_overlap=5,
        )
        index.setup()
        stats = index.reindex()

        assert stats["fts_indexed"] == 2
        assert stats["vectors_indexed"] > 0

        # Test semantic search
        results = index.search("code", mode="semantic", limit=5)
        assert isinstance(results, list)

        # Test hybrid search
        results = index.search("python", mode="hybrid", limit=5)
        assert isinstance(results, list)

        conn.close()

    def test_table_index_stats_with_vectors(self, tmp_path):
        """Test stats include vector counts."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(str(db_path))
        try:
            conn.enable_load_extension(True)
            import sqlite_vec

            sqlite_vec.load(conn)
        except Exception:
            conn.close()
            pytest.skip("sqlite-vec not available")

        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO docs VALUES (1, 'Test document')")
        conn.commit()

        class MockBackend:
            backend_name = "mock"
            dimensions = 8
            model_name = "mock"

            def embed(self, text):
                return [0.1] * 8

            def embed_batch(self, texts):
                return [[0.1] * 8 for _ in texts]

        index = TableIndex(
            conn=conn,
            table="docs",
            id_column="id",
            text_columns=["content"],
            embedding_backend=MockBackend(),
        )
        index.setup()
        index.reindex()

        stats = index.stats()

        assert stats["total_rows"] == 1
        assert stats["fts_indexed"] == 1
        assert "chunks_indexed" in stats
        assert "rows_with_vectors" in stats

        conn.close()
