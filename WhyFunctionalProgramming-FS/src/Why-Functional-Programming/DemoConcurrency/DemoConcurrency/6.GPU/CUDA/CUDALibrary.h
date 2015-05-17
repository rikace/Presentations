__device__ cudaError error;
 

template<class T>
struct List
{
	void* ptr;
	int length;
};

template<class T>
struct __RangeStruct
{
	int start;
	int end;
	unsigned int GetLength()
	{
		return (unsigned int)(end - start);
	}
};

template<class T>
struct __TupleStruct
{
	T x;
	T y;
};

template<class T>
__device__ __TupleStruct<T> newTuple(T x, T y)
{
	__TupleStruct<T> s;
	s.x = x;
	s.y = y;
	return s;
}

template <class T>
__device__ __RangeStruct<T> range(T x, T y)
{
	__RangeStruct<T> s;
	s.start = x;
	s.end = y;
	return s;
}

template <class T>
__device__ __RangeStruct<T> createSeq(__RangeStruct<T> range)
{
	return range;
}

template <class T>
__device__ List<T> toList(__RangeStruct<T> seq)
{
	unsigned int sizeT = (unsigned int)(sizeof(T));
	//unsigned int size = sizeT * seq.GetLength();
	List<T> t;	
	//error = cudaMalloc(&(t.ptr), size);
	return t;
}

template <class TInput, class TOutput>
__device__ List<TOutput> map(TInput (*fPtr)(TOutput), List<TInput> source)
{
	List<TOutput> t;
	t.length = 0;
	return t;
}
