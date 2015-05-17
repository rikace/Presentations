using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CSMisc
{
    public class Tree<TItem> where TItem : IComparable<TItem>
    {
        public Tree(TItem nodeValue)
        {
            this.NodeData = nodeValue;
            this.LeftTree = null;
            this.RightTree = null;
        }

        public void Insert(TItem newItem)
        {
            TItem currentNodeValue = this.NodeData;
            if (currentNodeValue.CompareTo(newItem) > 0)
            {
                if (this.LeftTree == null)
                {
                    this.LeftTree = new Tree<TItem>(newItem);
                }
                else
                {
                    this.LeftTree.Insert(newItem);
                }
            }
            else
            {
                if (this.RightTree == null)
                {
                    this.RightTree = new Tree<TItem>(newItem);
                }
                else
                {
                    this.RightTree.Insert(newItem);
                }
            }
        }

        public void WalkTree(Action<TItem> action)
        {
            if (this.LeftTree != null)
                this.LeftTree.WalkTree(action);

            action(this.NodeData);

            if (this.RightTree != null)
                this.RightTree.WalkTree(action);
        }

        public TItem NodeData { get; set; }
        public Tree<TItem> LeftTree { get; set; }
        public Tree<TItem> RightTree { get; set; }
    }
}