import os
import pandas as pd
import faiss
import numpy as np
import json
import time
from sentence_transformers import SentenceTransformer

# step 1:load baike data 

def load_baike_data(folder_path):
    data_frames = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            data_frames.append(df)
    baike_data = pd.concat(data_frames, ignore_index=True)
    return baike_data

# step 2: load eval data
def load_eval_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    eval_data = [json.loads(line) for line in lines]
    df_eval = pd.DataFrame(eval_data)
    return df_eval

# step 3: vectorize and save
def vectorize_questions(questions, model, save_path=None):
    vectors = model.encode(questions, convert_to_numpy=True).astype('float32')
    if save_path:
        np.save(save_path, vectors)  # 保存向量到指定路径
    return vectors
# step 4: construct FAISS index
def build_faiss_index(question_vectors):
    dimension = question_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(question_vectors)
    return index

# step 5 : search most similar question
def retrieve_similar_question(eval_question_vector, index, questions):
    distances,indices = index.search(eval_question_vector, 1)
    similar_idx = indices[0][0]
    #label = baike_data.iloc[similar_idx]['label']
    return questions[similar_idx]

# step 6 : main 
def main():
    # Step 6.1: 加载数据
    baike_folder_path = "med_ch"
    eval_file_path = "eval_data.jsonl"
    baike_data = load_baike_data(baike_folder_path)
    df_eval = load_eval_data(eval_file_path)

    # # Step 6.2: 向量化数据
    # baike_questions = baike_data['query'].tolist()
    # eval_questions = df_eval['question'].tolist()
    # print("正在向量化百科数据集")
    # baike_question_vectors = vectorize_questions(baike_questions, model)
    # print("正在向量化评估数据集...")
    # eval_question_vectors = vectorize_questions(eval_questions, model)
   

    # Step 6.2: 向量化数据（尝试加载向量化后的数据）
    print("初始化向量化模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    vectors_folder_path = "."  # 存放向量的文件夹就是项目根目录
    baike_vector_path = os.path.join(vectors_folder_path, "baike_question_vectors.npy")
    eval_vector_path = os.path.join(vectors_folder_path, "eval_question_vectors.npy")
    
    baike_questions = baike_data['query'].tolist()

    # 尝试加载百科问题的向量
    if os.path.exists(baike_vector_path):
        print("加载预计算的百科数据向量...")
        baike_question_vectors = np.load(baike_vector_path)
    else:
        print("正在向量化百科数据集...")
        #baike_questions = baike_data['query'].tolist()
        baike_question_vectors = vectorize_questions(baike_questions, model, save_path=baike_vector_path)

    # 尝试加载评估问题的向量
    eval_questions = df_eval['question'].tolist()

    if os.path.exists(eval_vector_path):
        print("加载预计算的评估数据向量...")
        eval_question_vectors = np.load(eval_vector_path)
    else:
        print("正在向量化评估数据集...")
        eval_question_vectors = vectorize_questions(eval_questions, model, save_path=eval_vector_path)
    
    # # Step 6.3: 构建FAISS索引
    print("正在构建FAISS索引...")
    index = build_faiss_index(baike_question_vectors)
    # Step 6.6: 对每个评估问题进行检索并保存结果
    total_start_time = time.time()
    results = []

    # 打开文件以保存结果
    with open("results.txt", "w", encoding="utf-8") as f:
        for i, eval_question_vector in enumerate(eval_question_vectors):
            eval_question_vector = np.expand_dims(eval_question_vector, axis=0)  # FAISS要求二维数组
            start_time = time.time()
            similar_question = retrieve_similar_question(eval_question_vector, index, baike_questions)
            end_time = time.time()

            # 打印结果
            result_text = (
                f"评估问题 {i + 1}: {eval_questions[i]}\n"
                f"最相似的问题: {similar_question}\n"
                f"检索时间: {end_time - start_time:.4f} 秒\n"
            )
            print(result_text)
            f.write(result_text + "\n")

            # 将结果保存到列表中，以备进一步分析
            results.append({
                "eval_question": eval_questions[i],
                "similar_question": similar_question,
                "retrieval_time": end_time - start_time
            })

    # 记录所有问题的总检索时间
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"所有问题的总检索时间: {total_elapsed_time:.4f} 秒")

    # 将总检索时间也保存到文件中
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"所有问题的总检索时间: {total_elapsed_time:.4f} 秒\n")

if __name__ == "__main__":
    main()
#     # Step 6.4: 对每个评估问题进行检索
#     # 记录所有问题检索的开始时间
#     total_start_time = time.time()
#     for i, eval_question_vector in enumerate(eval_question_vectors):
#         eval_question_vector = np.expand_dims(eval_question_vector, axis=0)  # FAISS要求二维数组
#         start_time = time.time()
#         similar_question  = retrieve_similar_question(eval_question_vector, index, baike_questions)
#         end_time = time.time()
#         # print
#         print(f"评估问题 {i + 1}: {eval_questions[i]}")
#         print(f"最相似的问题: {similar_question}")
#         #print(f"标签: {label}")
#         print(f"检索时间: {end_time - start_time:.4f} 秒\n")
#     # 记录所有问题检索的结束时间
#     total_end_time = time.time()
#     # 打印所有问题的总检索时间
#     total_elapsed_time = total_end_time - total_start_time
#     print(f"所有问题的总检索时间: {total_elapsed_time:.4f} 秒")
# if __name__ == "__main__":
#     main()
   





