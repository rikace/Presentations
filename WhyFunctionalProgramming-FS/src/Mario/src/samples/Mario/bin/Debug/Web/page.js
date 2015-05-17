$(document).ready(function () {
var Window__rgb$Int32__Int32__Int32_Int32_Int32_Int32, Window__position$Double_Double, Window__op_Dollar$String_String, Window__op_Dollar$Int32_Int32, Window__image$, Window__get_context$, Window__get_canvas$, Window__filled$String_String, Window__dimensions$, Window__context, Window__canvas, TupleSetTree_1_Int32__SetTree_1_Int32_, TupleInt32_SetTree_1_Int32_, TupleInt32_Int32, TupleDouble_Double_Double_Double, TupleDouble_Double, Set__Remove$Int32_Int32, Set__Empty$Int32_Int32, Set__Add$Int32_Int32, Set_1_Int32__get_Tree$Int32, Set_1_Int32__get_Empty$Int32, Set_1_Int32__get_Comparer$Int32, Set_1_Int32___ctor$Int32, Set_1_Int32__Remove$Int32, Set_1_Int32__Add$Int32, Set_1_IComparable__get_Tree$IComparable_, Set_1_IComparable__get_Comparer$IComparable_, Set_1_IComparable__Contains$IComparable_, SetTree_1_Int32__SetOneInt32, SetTree_1_Int32__SetNodeInt32, SetTree_1_Int32__SetEmptyInt32, SetTreeModule__tolerance, SetTreeModule__spliceOutSuccessor$Int32_Int32, SetTreeModule__remove$Int32_Int32, SetTreeModule__rebalance$Int32_Int32, SetTreeModule__mk$Int32_Int32, SetTreeModule__mem$IComparable_IComparable_, SetTreeModule__height$Int32_Int32, SetTreeModule__get_tolerance$, SetTreeModule__add$Int32_Int32, SetTreeModule__SetOne$Int32_Int32, SetTreeModule__SetNode$Int32_Int32, Program__walk$Int32_Int32, Program__step$, Program__render$, Program__physics$, Program__main$, Program__jump$Int32_Int32, Mario___ctor$, Lazy_1_Object__get_Value$Object_, Lazy_1_Object___ctor$Object_, Lazy_1_Object__Create$Object_, Keyboard__update$, Keyboard__keysPressed, Keyboard__init$, Keyboard__get_keysPressed$, Keyboard__code$, Keyboard__arrows$, GenericComparer_1_Int32___ctor$Int32;
  GenericComparer_1_Int32___ctor$Int32 = (function (unitVar0)
  {
    ;
  });
  Keyboard__arrows$ = (function (unitVar0)
  {
    return (new TupleInt32_Int32((Keyboard__code$(39) - Keyboard__code$(37)), (Keyboard__code$(38) - Keyboard__code$(40))));
  });
  Keyboard__code$ = (function (x)
  {
    if (Set_1_IComparable__Contains$IComparable_(Keyboard__keysPressed, x)) 
    {
      return 1;
    }
    else
    {
      return 0;
    };
  });
  Keyboard__get_keysPressed$ = (function ()
  {
    return Set__Empty$Int32_Int32();
  });
  Keyboard__init$ = (function (unitVar0)
  {
    (window.addEventListener("keydown", (function (e)
    {
      Keyboard__update$(e, true);
      return null;
    })));
    return (window.addEventListener("keyup", (function (e)
    {
      Keyboard__update$(e, false);
      return null;
    })));
  });
  Keyboard__update$ = (function (e, pressed)
  {
    var keyCode = (e.keyCode);
    var _7;
    if (pressed) 
    {
      _7 = (function (value)
      {
        return (function (set)
        {
          return Set__Add$Int32_Int32(value, set);
        });
      });
    }
    else
    {
      _7 = (function (value)
      {
        return (function (set)
        {
          return Set__Remove$Int32_Int32(value, set);
        });
      });
    };
    var op = _7;
    Keyboard__keysPressed = op(keyCode)(Keyboard__keysPressed);
  });
  Lazy_1_Object__Create$Object_ = (function (f)
  {
    return (new Lazy_1_Object___ctor$Object_(null, f));
  });
  Lazy_1_Object___ctor$Object_ = (function (value, factory)
  {
    this.factory = factory;
    this.isCreated = false;
    this.value_25 = value;
  });
  Lazy_1_Object__get_Value$Object_ = (function (x, unitVar1)
  {
    if ((!x.isCreated)) 
    {
      var _391;
      var _394;
      _391 = x.factory(_394);
      x.value_25 = _391;
      x.isCreated = true;
    }
    else
    {
      ;
    };
    return x.value_25;
  });
  Mario___ctor$ = (function (x, y, vx, vy, dir)
  {
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.dir = dir;
  });
  Mario___ctor$.prototype.CompareTo = (function (that)
  {
    var diff = 0.000000;
    var _diff = 0.000000;
    _diff = ((this.x < that.x) ? -1.000000 : ((this.x == that.x) ? 0.000000 : 1.000000));
    diff = _diff;
    if ((diff != 0.000000)) 
    {
      return diff;
    }
    else
    {
      var __diff = 0.000000;
      __diff = ((this.y < that.y) ? -1.000000 : ((this.y == that.y) ? 0.000000 : 1.000000));
      diff = __diff;
      if ((diff != 0.000000)) 
      {
        return diff;
      }
      else
      {
        var ___diff = 0.000000;
        ___diff = ((this.vx < that.vx) ? -1.000000 : ((this.vx == that.vx) ? 0.000000 : 1.000000));
        diff = ___diff;
        if ((diff != 0.000000)) 
        {
          return diff;
        }
        else
        {
          var ____diff = 0.000000;
          ____diff = ((this.vy < that.vy) ? -1.000000 : ((this.vy == that.vy) ? 0.000000 : 1.000000));
          diff = ____diff;
          if ((diff != 0.000000)) 
          {
            return diff;
          }
          else
          {
            var _____diff = 0.000000;
            _____diff = ((this.dir < that.dir) ? -1.000000 : ((this.dir == that.dir) ? 0.000000 : 1.000000));
            diff = _____diff;
            if ((diff != 0.000000)) 
            {
              return diff;
            }
            else
            {
              return 0.000000;
            };
          };
        };
      };
    };
  });
  Program__jump$Int32_Int32 = (function (_arg1, y, m)
  {
    if (((y > 0) && (m.y == 0.000000))) 
    {
      var vy = 5.000000;
      return (new Mario___ctor$(m.x, m.y, m.vx, vy, m.dir));
    }
    else
    {
      return m;
    };
  });
  Program__main$ = (function (unitVar0)
  {
    Keyboard__init$();
    var patternInput = Window__dimensions$();
    var w = patternInput.Items[0.000000];
    var h = patternInput.Items[1.000000];
    var update;
    update = (function (mario)
    {
      return (function (unitVar1)
      {
        var _417;
        var _418;
        var tupledArg = Keyboard__arrows$();
        var arg00_ = tupledArg.Items[0.000000];
        var arg01_ = tupledArg.Items[1.000000];
        _418 = (function (_mario)
        {
          return Program__step$(arg00_, arg01_, _mario);
        });
        _417 = _418(mario);
        var _mario = _417;
        Program__render$(w, h, _mario);
        return (function (value)
        {
          var ignored0 = value;
        })((window.setTimeout(update(_mario), (1000.000000 / 60.000000))));
      });
    });
    var mario = (new Mario___ctor$(0.000000, 0.000000, 0.000000, 0.000000, "right"));
    var _415;
    return update(mario)(_415);
  });
  Program__physics$ = (function (m)
  {
    var _498;
    if (((m.y + m.vy) > 0.000000)) 
    {
      _498 = (m.y + m.vy);
    }
    else
    {
      _498 = 0.000000;
    };
    return (new Mario___ctor$((m.x + m.vx), _498, m.vx, m.vy, m.dir));
  });
  Program__render$ = (function (w, h, mario)
  {
    var _590;
    var color = Window__rgb$Int32__Int32__Int32_Int32_Int32_Int32(174, 238, 238);
    _590 = (function (tupledArg)
    {
      var arg10_ = tupledArg.Items[0.000000];
      var arg11_ = tupledArg.Items[1.000000];
      var arg12_ = tupledArg.Items[2.000000];
      var arg13_ = tupledArg.Items[3.000000];
      return Window__filled$String_String(color, arg10_, arg11_, arg12_, arg13_);
    });
    _590((new TupleDouble_Double_Double_Double(0.000000, 0.000000, w, h)));
    var _662;
    var _color = Window__rgb$Int32__Int32__Int32_Int32_Int32_Int32(74, 163, 41);
    _662 = (function (tupledArg)
    {
      var arg10_ = tupledArg.Items[0.000000];
      var arg11_ = tupledArg.Items[1.000000];
      var arg12_ = tupledArg.Items[2.000000];
      var arg13_ = tupledArg.Items[3.000000];
      return Window__filled$String_String(_color, arg10_, arg11_, arg12_, arg13_);
    });
    _662((new TupleDouble_Double_Double_Double(0.000000, (h - 50.000000), w, 50.000000)));
    var _687;
    if ((mario.y > 0.000000)) 
    {
      _687 = "jump";
    }
    else
    {
      if ((mario.vx != 0.000000)) 
      {
        _687 = "walk";
      }
      else
      {
        _687 = "stand";
      };
    };
    var verb = _687;
    var _698;
    var tupledArg = (new TupleDouble_Double((((w / 2.000000) - 16.000000) + mario.x), (((h - 50.000000) - 31.000000) - mario.y)));
    var x = tupledArg.Items[0.000000];
    var y = tupledArg.Items[1.000000];
    _698 = (function (img)
    {
      return Window__position$Double_Double(x, y, img);
    });
    return _698((function (src)
    {
      return Window__image$(src);
    })(((("mario" + verb) + mario.dir) + ".gif")));
  });
  Program__step$ = (function (dir_0, dir_1, mario)
  {
    var dir = (new TupleInt32_Int32(dir_0, dir_1));
    var _523;
    var _524;
    var x = dir.Items[0.000000];
    var arg01_ = dir.Items[1.000000];
    _524 = (function (m)
    {
      return Program__walk$Int32_Int32(x, arg01_, m);
    });
    var _554;
    var _555;
    var arg00_ = dir.Items[0.000000];
    var y = dir.Items[1.000000];
    _555 = (function (m)
    {
      return Program__jump$Int32_Int32(arg00_, y, m);
    });
    _554 = _555(mario);
    _523 = _524(_554);
    return (function (m)
    {
      return Program__physics$(m);
    })(_523);
  });
  Program__walk$Int32_Int32 = (function (x, _arg1, m)
  {
    var vx = x;
    var _534;
    if ((x < 0)) 
    {
      _534 = "left";
    }
    else
    {
      if ((x > 0)) 
      {
        _534 = "right";
      }
      else
      {
        _534 = m.dir;
      };
    };
    var dir = _534;
    return (new Mario___ctor$(m.x, m.y, vx, m.vy, dir));
  });
  SetTreeModule__SetNode$Int32_Int32 = (function (x, l, r, h)
  {
    return (new SetTree_1_Int32__SetNodeInt32(x, l, r, h));
  });
  SetTreeModule__SetOne$Int32_Int32 = (function (n)
  {
    return (new SetTree_1_Int32__SetOneInt32(n));
  });
  SetTreeModule__add$Int32_Int32 = (function (comparer, k, t)
  {
    if ((t.Tag == 2.000000)) 
    {
      var k2 = t.Item;
      var c = comparer.Compare(k, k2);
      if ((c < 0)) 
      {
        return SetTreeModule__SetNode$Int32_Int32(k, (new SetTree_1_Int32__SetEmptyInt32()), t, 2);
      }
      else
      {
        if ((c == 0)) 
        {
          return t;
        }
        else
        {
          return SetTreeModule__SetNode$Int32_Int32(k, t, (new SetTree_1_Int32__SetEmptyInt32()), 2);
        };
      };
    }
    else
    {
      if ((t.Tag == 0.000000)) 
      {
        return SetTreeModule__SetOne$Int32_Int32(k);
      }
      else
      {
        var r = t.Item3;
        var l = t.Item2;
        var _k2 = t.Item1;
        var _c = comparer.Compare(k, _k2);
        if ((_c < 0)) 
        {
          return SetTreeModule__rebalance$Int32_Int32(SetTreeModule__add$Int32_Int32(comparer, k, l), _k2, r);
        }
        else
        {
          if ((_c == 0)) 
          {
            return t;
          }
          else
          {
            return SetTreeModule__rebalance$Int32_Int32(l, _k2, SetTreeModule__add$Int32_Int32(comparer, k, r));
          };
        };
      };
    };
  });
  SetTreeModule__get_tolerance$ = (function ()
  {
    return 2;
  });
  SetTreeModule__height$Int32_Int32 = (function (t)
  {
    if ((t.Tag == 2.000000)) 
    {
      return 1;
    }
    else
    {
      if ((t.Tag == 1.000000)) 
      {
        var h = t.Item4;
        return h;
      }
      else
      {
        return 0;
      };
    };
  });
  SetTreeModule__mem$IComparable_IComparable_ = (function (comparer, k, t)
  {
    if ((t.Tag == 2.000000)) 
    {
      var k2 = t.Item;
      return (comparer.Compare(k, k2) == 0);
    }
    else
    {
      if ((t.Tag == 0.000000)) 
      {
        return false;
      }
      else
      {
        var r = t.Item3;
        var l = t.Item2;
        var _k2 = t.Item1;
        var c = comparer.Compare(k, _k2);
        if ((c < 0)) 
        {
          return SetTreeModule__mem$IComparable_IComparable_(comparer, k, l);
        }
        else
        {
          return ((c == 0) || SetTreeModule__mem$IComparable_IComparable_(comparer, k, r));
        };
      };
    };
  });
  SetTreeModule__mk$Int32_Int32 = (function (l, k, r)
  {
    var matchValue = (new TupleSetTree_1_Int32__SetTree_1_Int32_(l, r));
    if ((matchValue.Items[0.000000].Tag == 0.000000)) 
    {
      if ((matchValue.Items[1.000000].Tag == 0.000000)) 
      {
        return SetTreeModule__SetOne$Int32_Int32(k);
      }
      else
      {
        var hl = SetTreeModule__height$Int32_Int32(l);
        var hr = SetTreeModule__height$Int32_Int32(r);
        var _140;
        if ((hl < hr)) 
        {
          _140 = hr;
        }
        else
        {
          _140 = hl;
        };
        var m = _140;
        return SetTreeModule__SetNode$Int32_Int32(k, l, r, (m + 1));
      };
    }
    else
    {
      var _hl = SetTreeModule__height$Int32_Int32(l);
      var _hr = SetTreeModule__height$Int32_Int32(r);
      var _154;
      if ((_hl < _hr)) 
      {
        _154 = _hr;
      }
      else
      {
        _154 = _hl;
      };
      var _m = _154;
      return SetTreeModule__SetNode$Int32_Int32(k, l, r, (_m + 1));
    };
  });
  SetTreeModule__rebalance$Int32_Int32 = (function (t1, k, t2)
  {
    var t1h = SetTreeModule__height$Int32_Int32(t1);
    var t2h = SetTreeModule__height$Int32_Int32(t2);
    if ((t2h > (t1h + SetTreeModule__tolerance))) 
    {
      if ((t2.Tag == 1.000000)) 
      {
        var t2r = t2.Item3;
        var t2l = t2.Item2;
        var t2k = t2.Item1;
        if ((SetTreeModule__height$Int32_Int32(t2l) > (t1h + 1))) 
        {
          if ((t2l.Tag == 1.000000)) 
          {
            var t2lr = t2l.Item3;
            var t2ll = t2l.Item2;
            var t2lk = t2l.Item1;
            return SetTreeModule__mk$Int32_Int32(SetTreeModule__mk$Int32_Int32(t1, k, t2ll), t2lk, SetTreeModule__mk$Int32_Int32(t2lr, t2k, t2r));
          }
          else
          {
            throw ("rebalance");
            return null;
          };
        }
        else
        {
          return SetTreeModule__mk$Int32_Int32(SetTreeModule__mk$Int32_Int32(t1, k, t2l), t2k, t2r);
        };
      }
      else
      {
        throw ("rebalance");
        return null;
      };
    }
    else
    {
      if ((t1h > (t2h + SetTreeModule__tolerance))) 
      {
        if ((t1.Tag == 1.000000)) 
        {
          var t1r = t1.Item3;
          var t1l = t1.Item2;
          var t1k = t1.Item1;
          if ((SetTreeModule__height$Int32_Int32(t1r) > (t2h + 1))) 
          {
            if ((t1r.Tag == 1.000000)) 
            {
              var t1rr = t1r.Item3;
              var t1rl = t1r.Item2;
              var t1rk = t1r.Item1;
              return SetTreeModule__mk$Int32_Int32(SetTreeModule__mk$Int32_Int32(t1l, t1k, t1rl), t1rk, SetTreeModule__mk$Int32_Int32(t1rr, k, t2));
            }
            else
            {
              throw ("rebalance");
              return null;
            };
          }
          else
          {
            return SetTreeModule__mk$Int32_Int32(t1l, t1k, SetTreeModule__mk$Int32_Int32(t1r, k, t2));
          };
        }
        else
        {
          throw ("rebalance");
          return null;
        };
      }
      else
      {
        return SetTreeModule__mk$Int32_Int32(t1, k, t2);
      };
    };
  });
  SetTreeModule__remove$Int32_Int32 = (function (comparer, k, t)
  {
    if ((t.Tag == 2.000000)) 
    {
      var k2 = t.Item;
      var c = comparer.Compare(k, k2);
      if ((c == 0)) 
      {
        return (new SetTree_1_Int32__SetEmptyInt32());
      }
      else
      {
        return t;
      };
    }
    else
    {
      if ((t.Tag == 1.000000)) 
      {
        var r = t.Item3;
        var l = t.Item2;
        var _k2 = t.Item1;
        var _c = comparer.Compare(k, _k2);
        if ((_c < 0)) 
        {
          return SetTreeModule__rebalance$Int32_Int32(SetTreeModule__remove$Int32_Int32(comparer, k, l), _k2, r);
        }
        else
        {
          if ((_c == 0)) 
          {
            var matchValue = (new TupleSetTree_1_Int32__SetTree_1_Int32_(l, r));
            if ((matchValue.Items[0.000000].Tag == 0.000000)) 
            {
              return r;
            }
            else
            {
              if ((matchValue.Items[1.000000].Tag == 0.000000)) 
              {
                return l;
              }
              else
              {
                var patternInput = SetTreeModule__spliceOutSuccessor$Int32_Int32(r);
                var sk = patternInput.Items[0.000000];
                var r_ = patternInput.Items[1.000000];
                return SetTreeModule__mk$Int32_Int32(l, sk, r_);
              };
            };
          }
          else
          {
            return SetTreeModule__rebalance$Int32_Int32(l, _k2, SetTreeModule__remove$Int32_Int32(comparer, k, r));
          };
        };
      }
      else
      {
        return t;
      };
    };
  });
  SetTreeModule__spliceOutSuccessor$Int32_Int32 = (function (t)
  {
    if ((t.Tag == 2.000000)) 
    {
      var k2 = t.Item;
      return (new TupleInt32_SetTree_1_Int32_(k2, (new SetTree_1_Int32__SetEmptyInt32())));
    }
    else
    {
      if ((t.Tag == 1.000000)) 
      {
        var r = t.Item3;
        var l = t.Item2;
        var _k2 = t.Item1;
        if ((l.Tag == 0.000000)) 
        {
          return (new TupleInt32_SetTree_1_Int32_(_k2, r));
        }
        else
        {
          var patternInput = SetTreeModule__spliceOutSuccessor$Int32_Int32(l);
          var l_ = patternInput.Items[1.000000];
          var k3 = patternInput.Items[0.000000];
          return (new TupleInt32_SetTree_1_Int32_(k3, SetTreeModule__mk$Int32_Int32(l_, _k2, r)));
        };
      }
      else
      {
        throw ("internal error: Map.spliceOutSuccessor");
        return null;
      };
    };
  });
  SetTree_1_Int32__SetEmptyInt32 = (function ()
  {
    this.Tag = 0.000000;
    this._CaseName = "SetEmpty";
  });
  SetTree_1_Int32__SetEmptyInt32.prototype.CompareTo = (function (that)
  {
    var diff = 0.000000;
    var _diff = 0.000000;
    _diff = ((this.Tag < that.Tag) ? -1.000000 : ((this.Tag == that.Tag) ? 0.000000 : 1.000000));
    diff = _diff;
    if ((diff != 0.000000)) 
    {
      return diff;
    }
    else
    {
      return 0.000000;
    };
  });
  SetTree_1_Int32__SetNodeInt32 = (function (Item1, Item2, Item3, Item4)
  {
    this.Tag = 1.000000;
    this._CaseName = "SetNode";
    this.Item1 = Item1;
    this.Item2 = Item2;
    this.Item3 = Item3;
    this.Item4 = Item4;
  });
  SetTree_1_Int32__SetNodeInt32.prototype.CompareTo = (function (that)
  {
    var diff = 0.000000;
    var _diff = 0.000000;
    _diff = ((this.Tag < that.Tag) ? -1.000000 : ((this.Tag == that.Tag) ? 0.000000 : 1.000000));
    diff = _diff;
    if ((diff != 0.000000)) 
    {
      return diff;
    }
    else
    {
      var __diff = 0.000000;
      __diff = ((this.Item1 < that.Item1) ? -1.000000 : ((this.Item1 == that.Item1) ? 0.000000 : 1.000000));
      diff = __diff;
      if ((diff != 0.000000)) 
      {
        return diff;
      }
      else
      {
        var ___diff = 0.000000;
        ___diff = this.Item2.CompareTo(that.Item2);
        diff = ___diff;
        if ((diff != 0.000000)) 
        {
          return diff;
        }
        else
        {
          var ____diff = 0.000000;
          ____diff = this.Item3.CompareTo(that.Item3);
          diff = ____diff;
          if ((diff != 0.000000)) 
          {
            return diff;
          }
          else
          {
            var _____diff = 0.000000;
            _____diff = ((this.Item4 < that.Item4) ? -1.000000 : ((this.Item4 == that.Item4) ? 0.000000 : 1.000000));
            diff = _____diff;
            if ((diff != 0.000000)) 
            {
              return diff;
            }
            else
            {
              return 0.000000;
            };
          };
        };
      };
    };
  });
  SetTree_1_Int32__SetOneInt32 = (function (Item)
  {
    this.Tag = 2.000000;
    this._CaseName = "SetOne";
    this.Item = Item;
  });
  SetTree_1_Int32__SetOneInt32.prototype.CompareTo = (function (that)
  {
    var diff = 0.000000;
    var _diff = 0.000000;
    _diff = ((this.Tag < that.Tag) ? -1.000000 : ((this.Tag == that.Tag) ? 0.000000 : 1.000000));
    diff = _diff;
    if ((diff != 0.000000)) 
    {
      return diff;
    }
    else
    {
      var __diff = 0.000000;
      __diff = ((this.Item < that.Item) ? -1.000000 : ((this.Item == that.Item) ? 0.000000 : 1.000000));
      diff = __diff;
      if ((diff != 0.000000)) 
      {
        return diff;
      }
      else
      {
        return 0.000000;
      };
    };
  });
  Set_1_IComparable__Contains$IComparable_ = (function (s, x)
  {
    return SetTreeModule__mem$IComparable_IComparable_(Set_1_IComparable__get_Comparer$IComparable_(s), x, Set_1_IComparable__get_Tree$IComparable_(s));
  });
  Set_1_IComparable__get_Comparer$IComparable_ = (function (set, unitVar1)
  {
    return set.comparer_479;
  });
  Set_1_IComparable__get_Tree$IComparable_ = (function (set, unitVar1)
  {
    return set.tree_483;
  });
  Set_1_Int32__Add$Int32 = (function (s, x)
  {
    return (new Set_1_Int32___ctor$Int32(Set_1_Int32__get_Comparer$Int32(s), SetTreeModule__add$Int32_Int32(Set_1_Int32__get_Comparer$Int32(s), x, Set_1_Int32__get_Tree$Int32(s))));
  });
  Set_1_Int32__Remove$Int32 = (function (s, x)
  {
    return (new Set_1_Int32___ctor$Int32(Set_1_Int32__get_Comparer$Int32(s), SetTreeModule__remove$Int32_Int32(Set_1_Int32__get_Comparer$Int32(s), x, Set_1_Int32__get_Tree$Int32(s))));
  });
  Set_1_Int32___ctor$Int32 = (function (comparer, tree)
  {
    this.comparer_479 = comparer;
    this.tree_483 = tree;
    this.serializedData = null;
  });
  Set_1_Int32__get_Comparer$Int32 = (function (set, unitVar1)
  {
    return set.comparer_479;
  });
  Set_1_Int32__get_Empty$Int32 = (function (unitVar0)
  {
    var comparer = (new GenericComparer_1_Int32___ctor$Int32());
    var _362;
    var impl;
    impl = comparer;
    _362 = {Compare: (function (x, y)
    {
      return (function (__, x, y)
      {
        var diff = 0.000000;
        diff = ((x < y) ? -1.000000 : ((x == y) ? 0.000000 : 1.000000));
        return diff;
      })(impl, x, y);
    })};
    return (new Set_1_Int32___ctor$Int32(_362, (new SetTree_1_Int32__SetEmptyInt32())));
  });
  Set_1_Int32__get_Tree$Int32 = (function (set, unitVar1)
  {
    return set.tree_483;
  });
  Set__Add$Int32_Int32 = (function (x, s)
  {
    return Set_1_Int32__Add$Int32(s, x);
  });
  Set__Empty$Int32_Int32 = (function ()
  {
    return Set_1_Int32__get_Empty$Int32();
  });
  Set__Remove$Int32_Int32 = (function (x, s)
  {
    return Set_1_Int32__Remove$Int32(s, x);
  });
  TupleDouble_Double = (function (Item0, Item1)
  {
    this.Items = [Item0, Item1];
    this.Items = [Item0, Item1];
  });
  TupleDouble_Double.prototype.CompareTo = (function (that)
  {
    var diff = 0.000000;
    var _diff = 0.000000;
    _diff = ((this.Items[0.000000] < that.Items[0.000000]) ? -1.000000 : ((this.Items[0.000000] == that.Items[0.000000]) ? 0.000000 : 1.000000));
    diff = _diff;
    if ((diff != 0.000000)) 
    {
      return diff;
    }
    else
    {
      var __diff = 0.000000;
      __diff = ((this.Items[1.000000] < that.Items[1.000000]) ? -1.000000 : ((this.Items[1.000000] == that.Items[1.000000]) ? 0.000000 : 1.000000));
      diff = __diff;
      if ((diff != 0.000000)) 
      {
        return diff;
      }
      else
      {
        return 0.000000;
      };
    };
  });
  TupleDouble_Double_Double_Double = (function (Item0, Item1, Item2, Item3)
  {
    this.Items = [Item0, Item1, Item2, Item3];
    this.Items = [Item0, Item1, Item2, Item3];
    this.Items = [Item0, Item1, Item2, Item3];
    this.Items = [Item0, Item1, Item2, Item3];
  });
  TupleDouble_Double_Double_Double.prototype.CompareTo = (function (that)
  {
    var diff = 0.000000;
    var _diff = 0.000000;
    _diff = ((this.Items[0.000000] < that.Items[0.000000]) ? -1.000000 : ((this.Items[0.000000] == that.Items[0.000000]) ? 0.000000 : 1.000000));
    diff = _diff;
    if ((diff != 0.000000)) 
    {
      return diff;
    }
    else
    {
      var __diff = 0.000000;
      __diff = ((this.Items[1.000000] < that.Items[1.000000]) ? -1.000000 : ((this.Items[1.000000] == that.Items[1.000000]) ? 0.000000 : 1.000000));
      diff = __diff;
      if ((diff != 0.000000)) 
      {
        return diff;
      }
      else
      {
        var ___diff = 0.000000;
        ___diff = ((this.Items[2.000000] < that.Items[2.000000]) ? -1.000000 : ((this.Items[2.000000] == that.Items[2.000000]) ? 0.000000 : 1.000000));
        diff = ___diff;
        if ((diff != 0.000000)) 
        {
          return diff;
        }
        else
        {
          var ____diff = 0.000000;
          ____diff = ((this.Items[3.000000] < that.Items[3.000000]) ? -1.000000 : ((this.Items[3.000000] == that.Items[3.000000]) ? 0.000000 : 1.000000));
          diff = ____diff;
          if ((diff != 0.000000)) 
          {
            return diff;
          }
          else
          {
            return 0.000000;
          };
        };
      };
    };
  });
  TupleInt32_Int32 = (function (Item0, Item1)
  {
    this.Items = [Item0, Item1];
    this.Items = [Item0, Item1];
  });
  TupleInt32_Int32.prototype.CompareTo = (function (that)
  {
    var diff = 0.000000;
    var _diff = 0.000000;
    _diff = ((this.Items[0.000000] < that.Items[0.000000]) ? -1.000000 : ((this.Items[0.000000] == that.Items[0.000000]) ? 0.000000 : 1.000000));
    diff = _diff;
    if ((diff != 0.000000)) 
    {
      return diff;
    }
    else
    {
      var __diff = 0.000000;
      __diff = ((this.Items[1.000000] < that.Items[1.000000]) ? -1.000000 : ((this.Items[1.000000] == that.Items[1.000000]) ? 0.000000 : 1.000000));
      diff = __diff;
      if ((diff != 0.000000)) 
      {
        return diff;
      }
      else
      {
        return 0.000000;
      };
    };
  });
  TupleInt32_SetTree_1_Int32_ = (function (Item0, Item1)
  {
    this.Items = [Item0, Item1];
    this.Items = [Item0, Item1];
  });
  TupleInt32_SetTree_1_Int32_.prototype.CompareTo = (function (that)
  {
    var diff = 0.000000;
    var _diff = 0.000000;
    _diff = ((this.Items[0.000000] < that.Items[0.000000]) ? -1.000000 : ((this.Items[0.000000] == that.Items[0.000000]) ? 0.000000 : 1.000000));
    diff = _diff;
    if ((diff != 0.000000)) 
    {
      return diff;
    }
    else
    {
      var __diff = 0.000000;
      __diff = this.Items[1.000000].CompareTo(that.Items[1.000000]);
      diff = __diff;
      if ((diff != 0.000000)) 
      {
        return diff;
      }
      else
      {
        return 0.000000;
      };
    };
  });
  TupleSetTree_1_Int32__SetTree_1_Int32_ = (function (Item0, Item1)
  {
    this.Items = [Item0, Item1];
    this.Items = [Item0, Item1];
  });
  TupleSetTree_1_Int32__SetTree_1_Int32_.prototype.CompareTo = (function (that)
  {
    var diff = 0.000000;
    var _diff = 0.000000;
    _diff = this.Items[0.000000].CompareTo(that.Items[0.000000]);
    diff = _diff;
    if ((diff != 0.000000)) 
    {
      return diff;
    }
    else
    {
      var __diff = 0.000000;
      __diff = this.Items[1.000000].CompareTo(that.Items[1.000000]);
      diff = __diff;
      if ((diff != 0.000000)) 
      {
        return diff;
      }
      else
      {
        return 0.000000;
      };
    };
  });
  Window__dimensions$ = (function (unitVar0)
  {
    return (new TupleDouble_Double((Lazy_1_Object__get_Value$Object_(Window__canvas).width), (Lazy_1_Object__get_Value$Object_(Window__canvas).height)));
  });
  Window__filled$String_String = (function (color, rect_0, rect_1, rect_2, rect_3)
  {
    var rect = (new TupleDouble_Double_Double_Double(rect_0, rect_1, rect_2, rect_3));
    var ctx = Lazy_1_Object__get_Value$Object_(Window__context);
    (ctx.fillStyle) = color;
    null;
    return (function (tupledArg)
    {
      var arg00 = tupledArg.Items[0.000000];
      var arg01 = tupledArg.Items[1.000000];
      var arg02 = tupledArg.Items[2.000000];
      var arg03 = tupledArg.Items[3.000000];
      return (ctx.fillRect(arg00, arg01, arg02, arg03));
    })(rect);
  });
  Window__get_canvas$ = (function ()
  {
    return Lazy_1_Object__Create$Object_((function (unitVar)
    {
      return (((window.document).getElementsByTagName("canvas"))[0]);
    }));
  });
  Window__get_context$ = (function ()
  {
    return Lazy_1_Object__Create$Object_((function (unitVar)
    {
      return (Lazy_1_Object__get_Value$Object_(Window__canvas).getContext("2d"));
    }));
  });
  Window__image$ = (function (src)
  {
    var image = (((window.document).getElementsByTagName("img"))[0]);
    if (((image.src).indexOf(src) == -1)) 
    {
      (image.src) = src;
      null;
    }
    else
    {
      ;
    };
    return image;
  });
  Window__op_Dollar$Int32_Int32 = (function (s, n)
  {
    return (s + n.toString());
  });
  Window__op_Dollar$String_String = (function (s, n)
  {
    return (s + n.toString());
  });
  Window__position$Double_Double = (function (x, y, img)
  {
    ((img.style).left) = (x.toString() + "px");
    null;
    var _735;
    var _736;
    var copyOfStruct = ((Lazy_1_Object__get_Value$Object_(Window__canvas).offsetTop) + y);
    _736 = copyOfStruct.toString();
    _735 = (_736 + "px");
    var _747;
    var _748;
    var copyOfStruct = ((Lazy_1_Object__get_Value$Object_(Window__canvas).offsetTop) + y);
    _748 = copyOfStruct.toString();
    _747 = (_748 + "px");
    ((img.style).top) = _735;
    return null;
  });
  Window__rgb$Int32__Int32__Int32_Int32_Int32_Int32 = (function (r, g, b)
  {
    return Window__op_Dollar$String_String(Window__op_Dollar$Int32_Int32(Window__op_Dollar$String_String(Window__op_Dollar$Int32_Int32(Window__op_Dollar$String_String(Window__op_Dollar$Int32_Int32("rgb(", r), ","), g), ","), b), ")");
  });
  SetTreeModule__tolerance = SetTreeModule__get_tolerance$();
  Keyboard__keysPressed = Keyboard__get_keysPressed$();
  Window__canvas = Window__get_canvas$();
  Window__context = Window__get_context$();
  return Program__main$()
});