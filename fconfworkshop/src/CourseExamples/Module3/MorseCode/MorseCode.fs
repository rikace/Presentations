module MorseCode

type MorseCode =
    | Node of string * MorseCode * MorseCode
    | Leaf of string
    | Null

let zeroNode = Leaf("0")
let nineNode = Leaf("9")
let dashNode = Node("", zeroNode, nineNode)
let eightNode = Leaf("8")
let dotNode = Node("", Null, eightNode)
let oNode = Node("O", dashNode, dotNode);
let qNode = Node("Q", Null, Null);
let sevenNode = Leaf("7");
let zNode = Node("Z", Null, sevenNode);
let gNode = Node("G", qNode, zNode);
let mNode = Node("M", oNode, gNode);
let yNode = Leaf("Y");
let cNode = Leaf("C");
let kNode = Node("K", yNode, cNode);
let xNode = Leaf("X");
let sixNode = Leaf("6");
let bNode = Node("B", Null, sixNode);
let dNode = Node("D", xNode, bNode);
let nNode = Node("N", kNode, dNode);
let tNode = Node("T", mNode, nNode);
let oneNode = Leaf("1");
let jNode = Node("J", oneNode, Null);
let pNode = Leaf("P");
let wNode = Node("W", jNode, pNode);
let lNode = Leaf("L");
let rNode = Node("R", Null, lNode);
let aNode = Node("A", wNode, rNode);
let twoNode = Leaf("2");
let udNode = Node("", twoNode, Null);
let fNode = Leaf("F");
let uNode = Node("U", udNode, fNode);
let threeNode = Leaf("3");
let vNode = Node("V", threeNode, Null);
let fourNode = Leaf("4");
let fiveNode = Leaf("5");
let hNode = Node("H", fourNode, fiveNode);
let sNode = Node("S", vNode, hNode);
let iNode = Node("I", uNode, sNode);
let eNode = Node("E", aNode, iNode);
let startNode = Node("", tNode, eNode);