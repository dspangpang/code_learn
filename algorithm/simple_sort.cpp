#include<iostream>
#include<cstring>

int Random(int start, int end){
    int dis = end - start;
    return rand() % dis + start;
}

void arr_init(int * arr, int size, int start, int end){
    
    for(int i = 0; i < size; i++){
        
        arr[i] = Random(start, end);
    }
}

void swap(int * arr, int a, int b){
    int tmp;
    tmp = arr[a];
    arr[a] = arr[b];
    arr[b] = tmp; 
}

void SelectionSort(int * arr, int arr_len){
    if(arr == NULL || arr_len < 2){
        return;
    }
    for(int i = 0; i < arr_len - 1; i++){  // 0 ～ N-1 数组中各个数的标号

        int minIdx = i;

        for(int j = i + 1 ; j < arr_len; j++){
            minIdx = arr[j] < arr[minIdx] ? j : minIdx;
        }
        swap(arr, i, minIdx);
    }
}

void BubbleSort(int * arr, int arr_len){
    if(arr == NULL || arr_len < 2){
        return;
    }
    for(int e = arr_len - 1; e > 0; e--){ 
        for(int i = 0; i < e; i++){
            if(arr[i] > arr[i + 1]){
                swap(arr, i, i + 1);
            }
        }
    }
}

void InsertionSort(int * arr, int arr_len){
    if(arr == NULL || arr_len < 2){
        return;
    }
    //0~0肯定有序
    //0～i有序

    for(int i = 1; i < arr_len; i++){  //0~i 要做到有序
        for(int j = i - 1; j >= 0 && arr[j] > arr[j + 1]; j--){
            swap(arr, j, j+1);
        }
    }
}

void three_sort_check(int * arr1, int * arr2, int * arr3, int size){

    for(int i = 0; i < size; i++){

        if((arr1[i] == arr2[i]) && (arr2[i] == arr3[i]));
        else{
            std::cout << "WRONG occured at" << i << std::endl;
            return; 
        }
    }

    std::cout << "OK" << std::endl; 
}

int process_arr_max(int * arr, int L, int R){

    if(L == R){
        return arr[L];
    }

    int mid = L + ((R-L) >> 1);           //防止溢出

    int RightMax = process_arr_max(arr, L, mid);
    int LeftMax = process_arr_max(arr, mid + 1, R);

    return (RightMax >= LeftMax)?(RightMax):(LeftMax);
}


int main(){

    const int ARRY_SIZE = 20;
    const int START = 0;
    const int END = 100;

    int arr_Select[ARRY_SIZE] = {0};
    int arr_Bubble[ARRY_SIZE] = {0};
    int arr_Insert[ARRY_SIZE] = {0};

    for(int i = 0; i < 20; i++){
        arr_init(arr_Select, ARRY_SIZE, START, END);
        memcpy(arr_Bubble, arr_Select, sizeof(int) * ARRY_SIZE);
        memcpy(arr_Insert, arr_Select, sizeof(int) * ARRY_SIZE);

        SelectionSort(arr_Select, ARRY_SIZE);
        BubbleSort(arr_Bubble, ARRY_SIZE);
        InsertionSort(arr_Insert, ARRY_SIZE);
        
        std::cout << "arr_Select largest is " << process_arr_max(arr_Select, 0, ARRY_SIZE - 1) << std::endl;
        std::cout << "arr_Bubble largest is " << process_arr_max(arr_Bubble, 0, ARRY_SIZE - 1) << std::endl;
        std::cout << "arr_Insert largest is " << process_arr_max(arr_Insert, 0, ARRY_SIZE - 1) << std::endl;

        three_sort_check(arr_Select, arr_Bubble, arr_Insert, ARRY_SIZE);
    }

    return 0;
}