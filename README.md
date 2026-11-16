# GF2 RAG Bridge

少女前线2：追放 剧情知识检索系统 — 云端检索服务 + SillyTavern 注入桥。

## 组成

- `server/` Node.js 检索服务（serve.js）
  - `POST /search` 混合检索（Embedding ∪ BM25 → RRF → 组去重 → Rerank）
  - `GET /context?id=<chunk>&before=N&after=N` 同组邻块续读（零 API 消耗）
  - `GET /health`
  - 索引文件（meta.json + vectors.f32）因体积不入库，需单独部署
- `extension/` SillyTavern 原生扩展（放 `public/scripts/extensions/third-party/gf2-rag-bridge/`）
  - 配置面板：RAG 地址、守门员/Embedding/Rerank 的 URL/Key/模型
  - `generate_interceptor` 生成前同步注入
- 剧情数据（`data/` 章节与角色目录、`worldbooks/` 世界书 JSON）不随仓库分发，需自备

## 部署

```bash
# 服务器（Node 18+）
node serve.js        # 默认 :5240，IDX=index 选择索引目录
# systemd 单元参考见部署记录
```

密钥不随仓库分发：首次启动生成 `config.json`，或经扩展面板保存配置。
