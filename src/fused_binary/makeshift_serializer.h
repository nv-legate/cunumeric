#include <iostream>
#include <vector>

class MakeshiftSerializer{
    
    public:
    MakeshiftSerializer(){
        size=4;
        raw.resize(size); 
        write_offset=0;
        read_offset=0;
    }

    template <typename T> void pack(T& arg) 
    {
        if (size<=write_offset+sizeof(T))
        {
            resize(sizeof(T));
        }
        raw[write_offset] = *reinterpret_cast<int8_t*>(&arg);
        write_offset+=sizeof(T);
    }

    template <typename T> void pack(T&& arg) 
    {
        T copy = arg;
        pack(copy);
    }

    template <typename T> T read() 
    {
        if (read_offset<write_offset)
        {
            T datum = *reinterpret_cast<T*>(raw.data()+read_offset);
            read_offset+=sizeof(T);
            return datum;
        }
        else{
            std::cout<<"finished reading buffer"<<std::endl;
            return NULL;
        }
    }

    void resize(size_t argSize){
        while(size<=write_offset+argSize)
        {
            //std::cout<<"resizing from "<<size<<" to "<<2*size<<std::endl; 
            size=2*size;
            raw.resize(size);
        }
    }

    void reset_reader(){
        read_offset=0;
    }

    private: 
    size_t size;
    int read_offset;
    int write_offset;
    std::vector<int8_t> raw;
};
/*
int main(){
    MakeshiftSerializer ms;
    int a=3; 
    char g='a'; 
    ms.pack<int>(a);
    ms.pack<char>(g);
    ms.pack<int>(a);
    ms.pack<char>(g);
    std::cout<<ms.read<int>()<<std::endl;;
    std::cout<<ms.read<char>()<<std::endl;;
    std::cout<<ms.read<int>()<<std::endl;;
    std::cout<<ms.read<char>()<<std::endl;;
    std::cout<<ms.read<int>()<<std::endl;;
    ms.reset_reader();
    std::cout<<ms.read<int>()<<std::endl;;
     
}*/
