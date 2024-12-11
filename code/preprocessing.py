# 定义三个文件的路径
file1_path = "sampleData/ankeny-sample.txt"
file2_path = "sampleData/cadastre-sample.txt"
file3_path = "sampleData/Haolong-sample.txt"
output_path = "sampleData/sample.txt"

# 读取每个文件的内容，并写入到新的文件中
with open(output_path, "w") as output_file:
    # 读取第一个文件并写入
    with open(file1_path, "r") as file1:
        output_file.write(file1.read())

    # 读取第二个文件并写入
    with open(file2_path, "r") as file2:
        output_file.write(file2.read())

    # 读取第三个文件并写入
    with open(file3_path, "r") as file3:
        output_file.write(file3.read())

print("文件合并完成，结果已保存到 merged_output.txt")
