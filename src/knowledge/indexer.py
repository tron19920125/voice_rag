"""
Azure AI Search 索引管理模块
负责创建索引、上传文档、管理向量搜索配置
"""

from typing import List, Dict, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)


class IndexManager:
    """Azure AI Search索引管理器"""

    def __init__(self, endpoint: str, api_key: str, index_name: str = "customer-service-kb"):
        """
        Args:
            endpoint: Azure AI Search端点
            api_key: API密钥
            index_name: 索引名称
        """
        self.endpoint = endpoint
        self.credential = AzureKeyCredential(api_key)
        self.index_name = index_name

        # 创建客户端
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        self.search_client = None  # 延迟初始化

    def create_index(self, vector_dimensions: int = 1024, force_recreate: bool = False) -> bool:
        """
        创建索引（支持文档+QA混合存储）

        Args:
            vector_dimensions: 向量维度（默认1024，对应bge-m3）
            force_recreate: 强制重建索引

        Returns:
            是否创建成功
        """
        # 检查索引是否已存在
        existing_indexes = [idx.name for idx in self.index_client.list_indexes()]

        if self.index_name in existing_indexes:
            if force_recreate:
                print(f"删除现有索引: {self.index_name}")
                self.index_client.delete_index(self.index_name)
            else:
                print(f"索引已存在: {self.index_name}")
                self._init_search_client()
                return True

        # 定义字段
        fields = [
            # 主键
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),

            # 类型字段（qa/document）
            SimpleField(
                name="type",
                type=SearchFieldDataType.String,
                filterable=True,
            ),

            # 文本字段（用于关键词搜索）
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                analyzer_name="zh-Hans.microsoft",  # 中文分词
            ),

            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                analyzer_name="zh-Hans.microsoft",
            ),

            # 向量字段（用于语义搜索）
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name="vector-profile",
            ),

            # QA专用字段
            SearchableField(
                name="question",
                type=SearchFieldDataType.String,
                analyzer_name="zh-Hans.microsoft",
            ),

            SimpleField(
                name="answer",
                type=SearchFieldDataType.String,
            ),

            # 元数据（简化为String，存储JSON）
            SimpleField(
                name="metadata",
                type=SearchFieldDataType.String,
            ),
        ]

        # 配置向量搜索
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config",
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine",
                    },
                )
            ],
        )

        # 配置语义搜索（可选）
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")],
            ),
        )

        semantic_search = SemanticSearch(configurations=[semantic_config])

        # 创建索引
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

        try:
            self.index_client.create_index(index)
            print(f"索引创建成功: {self.index_name}")
            self._init_search_client()
            return True
        except Exception as e:
            print(f"索引创建失败: {e}")
            return False

    def upload_documents(self, documents: List[Dict], batch_size: int = 100) -> Dict:
        """
        批量上传文档

        Args:
            documents: 文档列表，每个文档需包含：
                - id: 唯一ID
                - type: "document" 或 "qa"
                - content: 文本内容
                - content_vector: 向量（List[float]）
                - title: 标题
                - question: 问题（QA专用，可选）
                - answer: 答案（QA专用，可选）
                - metadata: 元数据（Dict，会转为JSON字符串）
            batch_size: 批处理大小

        Returns:
            上传统计信息
        """
        if not self.search_client:
            self._init_search_client()

        # 预处理文档（转换metadata为JSON字符串）
        import json
        processed_docs = []
        for doc in documents:
            processed_doc = doc.copy()
            if "metadata" in processed_doc and isinstance(processed_doc["metadata"], dict):
                processed_doc["metadata"] = json.dumps(processed_doc["metadata"], ensure_ascii=False)
            processed_docs.append(processed_doc)

        # 批量上传
        total = len(processed_docs)
        success_count = 0
        fail_count = 0

        for i in range(0, total, batch_size):
            batch = processed_docs[i:i+batch_size]
            try:
                result = self.search_client.upload_documents(documents=batch)
                success_count += len([r for r in result if r.succeeded])
                fail_count += len([r for r in result if not r.succeeded])
            except Exception as e:
                print(f"批次上传失败 ({i}-{i+len(batch)}): {e}")
                fail_count += len(batch)

        stats = {
            "total": total,
            "success": success_count,
            "failed": fail_count,
        }

        print(f"上传完成: {stats}")
        return stats

    def delete_documents(self, document_ids: List[str]) -> int:
        """
        删除文档

        Args:
            document_ids: 文档ID列表

        Returns:
            删除成功的数量
        """
        if not self.search_client:
            self._init_search_client()

        try:
            result = self.search_client.delete_documents(
                documents=[{"id": doc_id} for doc_id in document_ids]
            )
            success_count = len([r for r in result if r.succeeded])
            print(f"删除文档: {success_count}/{len(document_ids)} 成功")
            return success_count
        except Exception as e:
            print(f"删除文档失败: {e}")
            return 0

    def get_document_count(self) -> int:
        """获取索引中的文档总数"""
        if not self.search_client:
            self._init_search_client()

        try:
            result = self.search_client.search(
                search_text="*",
                include_total_count=True,
                top=0,
            )
            return result.get_count()
        except Exception as e:
            print(f"获取文档数失败: {e}")
            return 0

    def index_exists(self) -> bool:
        """检查索引是否存在"""
        existing_indexes = [idx.name for idx in self.index_client.list_indexes()]
        return self.index_name in existing_indexes

    def _init_search_client(self):
        """初始化搜索客户端"""
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )
