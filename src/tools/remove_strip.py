import os  # 用于操作文件和目录


def process_txt_files(directory):
    """
    处理指定目录下所有TXT文件，去除每一行的前后空格
    :param directory: 目标目录路径（绝对路径或相对路径均可）
    """
    # 1. 检查目录是否存在，不存在则提示并退出
    if not os.path.exists(directory):
        print(f"错误：目录 '{directory}' 不存在，请检查路径是否正确！")
        return

    # 2. 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 只处理后缀为 .txt 的文件
        if filename.endswith(".txt"):
            # 拼接完整的文件路径（避免因工作目录不同导致的路径错误）
            file_path = os.path.join(directory, filename)
            # 拼接备份文件路径（原文件名 + _backup 后缀）
            backup_path = os.path.join(directory, f"{os.path.splitext(filename)[0]}_backup.txt")

            print(f"\n正在处理文件：{filename}")

            # 3. 备份原始文件（先复制再处理，确保原始数据安全）
            try:
                with open(file_path, "r", encoding="utf-8") as original_file, \
                        open(backup_path, "w", encoding="utf-8") as backup_file:
                    # 直接将原始文件内容复制到备份文件
                    backup_file.write(original_file.read())
                print(f"✅ 已创建备份文件：{os.path.basename(backup_path)}")
            except Exception as e:
                print(f"❌ 备份文件 {filename} 失败：{str(e)}")
                continue  # 备份失败则跳过该文件，避免覆盖原始数据

            # 4. 读取原始文件内容，去除每一行前后空格
            processed_lines = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        # 去除当前行前后的空格/制表符，保留行内中间空格
                        cleaned_line = line.strip()
                        # 将处理后的行加入列表（需手动添加换行符，strip()会移除原换行）
                        processed_lines.append(cleaned_line + "\n")
                print(f"✅ 成功读取并处理 {filename} 的内容")
            except Exception as e:
                print(f"❌ 读取文件 {filename} 失败：{str(e)}")
                continue

            # 5. 将处理后的内容写回原文件（覆盖原内容，已备份故安全）
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(processed_lines)
                print(f"✅ 已将处理后的内容写回 {filename}")
            except Exception as e:
                print(f"❌ 写入文件 {filename} 失败：{str(e)}")
                # 若写入失败，可恢复备份（可选，此处简化为提示）
                print(f"提示：可通过备份文件 {os.path.basename(backup_path)} 恢复原始数据")

    print("\n" + "=" * 50)
    print("所有TXT文件处理完成！（未处理非TXT文件）")
    print("注意：备份文件保留在原目录，确认处理结果无误后可自行删除备份。")


# ------------------- 请在这里修改目标目录路径 -------------------
# 示例1：相对路径（脚本所在文件夹下的 "txt_files" 目录，需提前创建）
target_directory = r"E:\fish\comp\test\test\submit"
# 示例2：绝对路径（Windows系统，注意路径用双反斜杠或单斜杠）
# target_directory = "C:\\Users\\YourName\\Desktop\\my_txts"
# 示例3：绝对路径（Mac/Linux系统）
# target_directory = "/Users/YourName/Desktop/my_txts"
# -------------------------------------------------------------

# 执行主函数
if __name__ == "__main__":
    process_txt_files(target_directory)