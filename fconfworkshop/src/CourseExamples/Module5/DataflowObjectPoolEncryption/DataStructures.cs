using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataflowObjectPoolEncryption
{
    public struct CompressingDetails
    {
        public Chunk Bytes { get; set; }
        public int Sequence { get; set; }
        public Chunk ChunkSize { get; set; }
    }

    public struct CompressedDetails
    {
        public Chunk Bytes { get; set; }
        public int Sequence { get; set; }
        public Chunk ChunkSize { get; set; }
        public Chunk CompressedDataSize { get; set; }
        public bool IsProcessed { get; set; }
    }
    public struct EncryptDetails
    {
        public Chunk Bytes { get; set; }
        public int Sequence { get; set; }
        public Chunk EncryptedDataSize { get; set; }
        public bool IsProcessed { get; set; }
    }

    public struct DecompressionDetails
    {

        public Chunk Bytes { get; set; }
        public int Sequence { get; set; }
        public long ChunkSize { get; set; }
        public bool IsProcessed { get; set; }
    }
    public struct DecryptDetails
    {
        public Chunk Bytes { get; set; }
        public int Sequence { get; set; }
        public Chunk EncryptedDataSize { get; set; }
    }
}
