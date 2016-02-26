using Microsoft.VisualStudio.TestTools.UnitTesting;

using Sodium;

namespace Tests.sodium
{
  [TestClass]
  public class NodeTester
  {
    [TestMethod]
    public void TestNode()
    {
      var a = new Node(0);
      var b = new Node(1);
      a.LinkTo(b);
            Assert.Equals(a, b);
    }
  }
}