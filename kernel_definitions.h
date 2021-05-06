//
// Created by tobias on 04.05.21.
//

#ifndef CUDA_KERNEL_KERNEL_DEFINITIONS_H
#define CUDA_KERNEL_KERNEL_DEFINITIONS_H

template<size_t S>
class BinaryArray
{
public:
    constexpr size_t Size() const { return S; }
    constexpr size_t actualSize() const { return (S + 31)/32; }

    void set(unsigned int index, bool value) {
        int bit_index = index % 32;
        m_data[index/32] = (m_data[index/32] & ~(1UL << bit_index)) | (value << bit_index);
    }

    bool get(unsigned int index) {
        int bit_index = index % 32;
        return (m_data[index/32] >> bit_index) & 1UL;
    }
private:
    unsigned int m_data[(S+31)/32];
};

template <typename T>
void result_to_csv(T *result, const int N, const int M, const char *filename) {
    std::ofstream ResultFile;
    ResultFile.open(filename);

    for (int k = 0; k < N; k++) {
        for (int l = 0; l < M; l++) {

            ResultFile << result[k * M + l];

            if (l < M-1) {
                ResultFile << ",";
            }
            else {
                ResultFile << "\n";
            }
        }
    }

    ResultFile.close();
}

template<typename T>
void read_csv_to_array(T *result_array, const int N, const int M, const char *filename) {
    std::ifstream target_file;
    target_file.open(filename);

    char temp_for_delimiter;
    std::string file_line;

    if (target_file.good()) {
        for (int j = 0; j < N; ++j) {
            std::getline(target_file, file_line);
            std::stringstream iss(file_line);
            iss >> result_array[j * M];

            for (int i = 1; i < M; ++i) {
                iss >> temp_for_delimiter >> result_array[j * M + i];
            }
        }
    }

    target_file.close();

}

template <size_t S>
void result_to_csv_binary(BinaryArray<S> *result, const int N, const int M, const char *filename) {
    std::ofstream ResultFile;
    ResultFile.open(filename);

    for (int k = 0; k < N; k++) {
        for (int l = 0; l < M; l++) {

            ResultFile << result->get(k * M + l);

            if (l < M-1) {
                ResultFile << ",";
            }
            else {
                ResultFile << "\n";
            }
        }
    }

    ResultFile.close();
}


#endif //CUDA_KERNEL_KERNEL_DEFINITIONS_H
