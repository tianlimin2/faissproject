
## 项目说明

### 项目目标

  * **向量化:** 利用 Sentence Transformers 将医疗问诊数据映射到高维向量空间。
  * **索引构建:** 使用 FAISS 库构建高效的近邻搜索索引，实现对海量医疗问题的快速检索。
  * **相似性匹配:** 根据用户输入问题，在索引中找到语义上最相似的医疗问题。
  * **性能评估:** 评估模型在医疗场景下的准确性和效率。

### 数据

  * **医疗问诊数据:** 包含患者的症状描述、就诊科室等信息，以 CSV 格式存储。
  * **评估数据集:** 用于评估模型性能，包含问题和对应的标准答案。

### 系统流程

1.  **数据加载:** 从指定路径加载医疗问诊数据和评估数据集。
2.  **向量化:**
      * 使用 Sentence Transformers 将问题文本转换为密集向量。
      * 将向量保存到磁盘以加速后续操作。
3.  **索引构建:**
      * 利用 FAISS 库构建高效的索引，存储向量数据。
4.  **相似性搜索:**
      * 将待查询的问题向量化，在索引中查找最近邻。
5.  **结果输出:**
      * 将搜索结果（相似问题）保存为 JSON 格式。

### 代码结构

  * **`load_baike_data`:** 加载医疗问诊数据。
  * **`load_eval_data`:** 加载评估数据集。
  * **`vectorize_questions`:** 将问题文本向量化。
  * **`build_faiss_index`:** 构建 FAISS 索引。
  * **`retrieve_similar_questions`:** 执行相似性搜索。
  * **`main`:** 主函数，整合以上功能。

### 技术栈

  * **Sentence Transformers:** 用于文本编码。
  * **FAISS:** 用于构建高效的近邻搜索索引。
  * **Pandas:** 用于数据处理。
  * **NumPy:** 用于数值计算。

### 运行方式

1.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **配置参数:**
      * 修改代码中数据路径、模型参数等配置。
3.  **运行程序:**
    ```bash
    python your_script.py
    ```