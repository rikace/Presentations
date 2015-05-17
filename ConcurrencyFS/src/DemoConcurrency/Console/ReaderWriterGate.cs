using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Diagnostics;
using System.Globalization;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.Contracts;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using System.ComponentModel;

namespace Console
{
    internal sealed class CatalogEntry { }
    internal sealed class CatIDAndQuantity { }

    internal sealed class CatalogOrderSystem
    {
        private ReaderWriterGate m_gate = new ReaderWriterGate();

        public void UpdateCatalog(CatalogEntry[] catalogEntries)
        {
            // Perform any validation/pre-processing on catalogEntries...

            // Updating the catalog requires exclusive access to it.
            m_gate.BeginWrite(UpdateCatalog, catalogEntries,
               delegate(IAsyncResult result) { m_gate.EndWrite(result); }, null);
        }

        // The code in this method has exclusive access to the catalog.
        private Object UpdateCatalog(ReaderWriterGateReleaser r)
        {
            CatalogEntry[] catalogEntries = (CatalogEntry[])r.State;
            // Update the catalog with the new entries...

            // When this method returns, exclusive access is relinquished.
            return null;
        }


        public void BuyCatalogProducts(CatIDAndQuantity[] items)
        {
            // Buying products requires read access to the catalog.
            m_gate.BeginRead(BuyCatalogProducts, items, delegate(IAsyncResult result)
            {
                m_gate.EndRead(result);
            }, null);
        }

        // The code in this method has shared read access to the catalog.
        private Object BuyCatalogProducts(ReaderWriterGateReleaser r)
        {
            using (r)
            {
                CatIDAndQuantity[] items = (CatIDAndQuantity[])r.State;
                foreach (CatIDAndQuantity item in items)
                {
                    // Process each catalog item to build customer's order...
                }
            } // When r is Disposed, read access is relinquished.

            // Save customer's order to database
            // Send customer e-mail confirming order
            return null;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////



    public delegate Object ReaderWriterGateCallback(ReaderWriterGateReleaser releaser);

    public enum ResourceLockOptions
    {
        /// <summary>
        /// None.
        /// </summary>
        None = 0x00000000,

        /// <summary>
        /// If specified, then the thread that acquires the lock must also 
        /// release the lock. No other thread can release the lock.
        /// </summary>
        AcquiringThreadMustRelease = 0x00000001,

        /// <summary>
        /// If specified, then this lock supports recursion.
        /// </summary>
        SupportsRecursion = 0x00000002,

        /// <summary>
        /// Indicates that this lock is really a mutual-exclusive lock allowing only one thread to enter into it at a time.
        /// </summary>
        IsMutualExclusive = 0x00000004,

#if DEADLOCK_DETECTION
      /// <summary>
      /// If specified, then deadlock detection does not apply to this kind of lock.
      /// </summary>
      ImmuneFromDeadlockDetection = unchecked((Int32)0x80000000)
#endif
    }


    [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1724:TypeNamesShouldNotMatchNamespaces")]
    public sealed class ReaderWriterGate : IDisposable
    {
        private ResourceLock m_syncLock = new MonitorResourceLock();

        /// <summary>
        /// Releases all resources associated with the ReaderWriterGate
        /// </summary>
        public void Dispose() { m_syncLock.Dispose(); }

        private enum ReaderWriterGateStates
        {
            Free = 0,
            OwnedByReaders = 1,
            OwnedByReadersAndWriterPending = 2,
            OwnedByWriter = 3,
            ReservedForWriter = 4
        }
        private ReaderWriterGateStates m_state = ReaderWriterGateStates.Free;
        private Int32 m_numReaders = 0;

        private Queue<ReaderWriterGateReleaser> m_qWriteRequests = new Queue<ReaderWriterGateReleaser>();
        private Queue<ReaderWriterGateReleaser> m_qReadRequests = new Queue<ReaderWriterGateReleaser>();

        /// <summary>
        /// Constructs a ReaderWriterGate object.
        /// </summary>
        public ReaderWriterGate() : this(false) { }

        /// <summary>
        /// Constructs a ReaderWriterGate object
        /// </summary>
        /// <param name="blockReadersUntilFirstWriteCompletes">Pass true to have readers block until the first writer has created the data that is being protected by the ReaderWriterGate.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Performance", "CA1805:DoNotInitializeUnnecessarily")]
        public ReaderWriterGate(Boolean blockReadersUntilFirstWriteCompletes)
        {
            m_state = blockReadersUntilFirstWriteCompletes ? ReaderWriterGateStates.ReservedForWriter : ReaderWriterGateStates.Free;
        }

        #region BeginWrite/EndWrite Members
        /// <summary>Initiates an asynchronous write operation.</summary>
        /// <param name="callback">The method that will perform the write operation.</param>
        /// <param name="asyncCallback">An optional asynchronous callback, to be called when the operation completes.</param>
        /// <returns>A System.IAsyncResult that represents the asynchronous operation, which could still be pending.</returns>
        public IAsyncResult BeginWrite(ReaderWriterGateCallback callback, AsyncCallback asyncCallback)
        {
            return BeginWrite(callback, null, asyncCallback, null);
        }

        /// <summary>Initiates an asynchronous write operation.</summary>
        /// <param name="callback">The method that will perform the write operation.</param>
        /// <param name="state">A value passed to the callback method.</param>
        /// <param name="asyncCallback">An optional asynchronous callback, to be called when the operation completes.</param>
        /// <param name="asyncState">A user-provided object that distinguishes this particular asynchronous operation request from other requests.</param>
        /// <returns>A System.IAsyncResult that represents the asynchronous operation, which could still be pending.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Reliability", "CA2000:Dispose objects before losing scope")]
        public IAsyncResult BeginWrite(ReaderWriterGateCallback callback, Object state,
          AsyncCallback asyncCallback, Object asyncState)
        {
            AsyncResult<Object> ar = new AsyncResult<Object>(asyncCallback, asyncState);
            ReaderWriterGateReleaser releaser = new ReaderWriterGateReleaser(callback, this, false, state, ar);
            m_syncLock.Enter(true);
            switch (m_state)
            {
                case ReaderWriterGateStates.Free:             // If Free "RFW -> OBW, invoke, return
                case ReaderWriterGateStates.ReservedForWriter:
                    m_state = ReaderWriterGateStates.OwnedByWriter;
                    ThreadPool.QueueUserWorkItem(releaser.Invoke);
                    break;

                case ReaderWriterGateStates.OwnedByReaders:   // If OBR | OBRAWP -> OBRAWP, queue, return
                case ReaderWriterGateStates.OwnedByReadersAndWriterPending:
                    m_state = ReaderWriterGateStates.OwnedByReadersAndWriterPending;
                    m_qWriteRequests.Enqueue(releaser);
                    break;

                case ReaderWriterGateStates.OwnedByWriter:   // If OBW, queue, return
                    m_qWriteRequests.Enqueue(releaser);
                    break;
            }
            m_syncLock.Leave();
            return ar;
        }

        /// <summary>Returns the result of the asynchronous operation.</summary>
        /// <param name="result">The reference to the pending asynchronous operation to finish.</param>
        /// <returns>Whatever the write callback method returns.</returns>
        public Object EndWrite(IAsyncResult result)
        {
            if (result == null) throw new ArgumentNullException("result");
            return ((AsyncResult<Object>)result).EndInvoke();
        }
        #endregion


        #region BeginRead/EndWrite Members
        /// <summary>Initiates an asynchronous read operation.</summary>
        /// <param name="callback">The method that will perform the read operation.</param>
        /// <param name="asyncCallback">An optional asynchronous callback, to be called when the operation completes.</param>
        /// <returns>A System.IAsyncResult that represents the asynchronous operation, which could still be pending.</returns>
        public IAsyncResult BeginRead(ReaderWriterGateCallback callback, AsyncCallback asyncCallback)
        {
            return BeginRead(callback, null, asyncCallback, null);
        }

        /// <summary>Initiates an asynchronous read operation.</summary>
        /// <param name="callback">The method that will perform the read operation.</param>
        /// <param name="state">A value passed to the callback method.</param>
        /// <param name="asyncCallback">An optional asynchronous callback, to be called when the operation completes.</param>
        /// <param name="asyncState">A user-provided object that distinguishes this particular asynchronous operation request from other requests.</param>
        /// <returns>A System.IAsyncResult that represents the asynchronous operation, which could still be pending.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Reliability", "CA2000:Dispose objects before losing scope")]
        public IAsyncResult BeginRead(ReaderWriterGateCallback callback, Object state,
           AsyncCallback asyncCallback, Object asyncState)
        {
            AsyncResult<Object> ar = new AsyncResult<Object>(asyncCallback, asyncState);
            ReaderWriterGateReleaser releaser = new ReaderWriterGateReleaser(callback, this, true, state, ar);
            m_syncLock.Enter(true);
            switch (m_state)
            {
                case ReaderWriterGateStates.Free:   // If Free | OBR -> OBR, NR++, invoke, return
                case ReaderWriterGateStates.OwnedByReaders:
                    m_state = ReaderWriterGateStates.OwnedByReaders;
                    m_numReaders++;
                    ThreadPool.QueueUserWorkItem(releaser.Invoke);
                    break;

                case ReaderWriterGateStates.OwnedByWriter:   // If OBW | OBRAWP | RFW, queue, return
                case ReaderWriterGateStates.OwnedByReadersAndWriterPending:
                case ReaderWriterGateStates.ReservedForWriter:
                    m_qReadRequests.Enqueue(releaser);
                    break;
            }
            m_syncLock.Leave();
            return ar;
        }

        /// <summary>Returns the result of the asynchronous read operation.</summary>
        /// <param name="result">The reference to the pending asynchronous operation to finish.</param>
        /// <returns>Whatever the read callback method returns.</returns>
        public Object EndRead(IAsyncResult result)
        {
            if (result == null) throw new ArgumentNullException("result");
            return ((AsyncResult<Object>)result).EndInvoke();
        }
        #endregion


        #region Helper Classes and Methods
        internal void Release(Boolean reader)
        {
            m_syncLock.Enter(true);
            // If writer or last reader, the lock is being freed
            Boolean freeing = reader ? (--m_numReaders == 0) : true;
            if (freeing)
            {
                // Wake up a writer, or all readers, or set to free
                if (m_qWriteRequests.Count > 0)
                {
                    // A writer is queued, invoke it
                    m_state = ReaderWriterGateStates.OwnedByWriter;
                    ThreadPool.QueueUserWorkItem(m_qWriteRequests.Dequeue().Invoke);
                }
                else if (m_qReadRequests.Count > 0)
                {
                    // Reader(s) are queued, invoke all of them
                    m_state = ReaderWriterGateStates.OwnedByReaders;
                    m_numReaders = m_qReadRequests.Count;
                    while (m_qReadRequests.Count > 0)
                    {
                        ThreadPool.QueueUserWorkItem(m_qReadRequests.Dequeue().Invoke);
                    }
                }
                else
                {
                    // No writers or readers, free the gate
                    m_state = ReaderWriterGateStates.Free;
                }
            }
            m_syncLock.Leave();
        }
        #endregion
    }

    public sealed class ReaderWriterGateReleaser : IDisposable
    {
        [Flags]
        private enum ReleaserFlags
        {
            Reader = 0x0000000,
            Writer = 0x0000001,
            Completed = 0x00000002,
        }

        private ReaderWriterGateCallback m_callback;
        private ReaderWriterGate m_gate;
        private ReleaserFlags m_flags;
        private Object m_state;
        private AsyncResult<Object> m_asyncResult;
        private Object m_resultValue;

        internal ReaderWriterGateReleaser(ReaderWriterGateCallback callback, ReaderWriterGate gate,
           Boolean reader, Object state, AsyncResult<Object> ar)
        {

            m_callback = callback;
            m_gate = gate;
            m_flags = reader ? ReleaserFlags.Reader : ReleaserFlags.Writer;
            m_state = state;
            m_asyncResult = ar;
        }

        /// <summary>Returns the ReaderWriterGate used to initiate the read/write operation.</summary>
        public ReaderWriterGate Gate
        {
            get { return m_gate; }
        }

        /// <summary>Returns the state that was passed with the read/write operation request.</summary>
        public Object State
        {
            get { return m_state; }
        }

        /// <summary>Allows the read/write callback method to return a value from EndRead/EndWrite.</summary>
        public Object ResultValue
        {
            get { return m_resultValue; }
            set { m_resultValue = value; }
        }

        [SuppressMessage("Microsoft.Design", "CA1031:DoNotCatchGeneralExceptionTypes",
          Justification = "OK because exception will be thrown by EndInvoke.")]
        internal void Invoke(Object o)
        {
            // Called via ThreadPool.QueueUserWorkItem; argument is ignored
            try
            {
                Complete(null, m_callback(this), false);
            }
            catch (Exception e)
            {
                if (!Complete(e, null, false)) throw;
            }
        }

        /// <summary>Releases the ReaderWriterGate to that other read/write operations may start.</summary>
        public void Dispose() { Release(false); }

        /// <summary>Releases the ReaderWriterGate so that other read/write operations may start.</summary>
        public void Release(Boolean completeOnReturn = false) { Complete(null, ResultValue, completeOnReturn); }

        private Boolean Complete(Exception exception, Object resultValue, Boolean completeOnReturn)
        {
            Boolean success = false;  // Assume this call fails

            if (m_gate != null)
            {
                // Gate not already released; release it
                Boolean reader = (m_flags & ReleaserFlags.Writer) == 0;
                m_gate.Release(reader);  // Release the gate
                m_gate = null; // Mark as complete so we don't complete again
                success = true;
            }

            // If completeOnReturn is true, then the gate is being released explicitly (via Release) and we should NOT complete the operation

            // If we're returning and we're released the gate, then indicate that the operation is complete
            if (completeOnReturn) { success = true; }
            else
            {
                // Else we should complete this operation if we didn't do it already
                if ((m_flags & ReleaserFlags.Completed) == 0)
                {
                    m_flags |= ReleaserFlags.Completed;
                    // Signal the completion with the exception or the ResultValue
                    if (exception != null) m_asyncResult.SetAsCompleted(exception, false);
                    else m_asyncResult.SetAsCompleted(resultValue, false);
                    success = true;
                }
            }
            return success;   // This call to complete succeeded
        }
    }


    public sealed class MonitorResourceLock : ResourceLock
    {
        private readonly Object m_lock;

        /// <summary>
        /// Constructs an instance on the MonitorResourceLock.
        /// </summary>
        public MonitorResourceLock()
            : base(ResourceLockOptions.AcquiringThreadMustRelease | ResourceLockOptions.IsMutualExclusive | ResourceLockOptions.SupportsRecursion) { m_lock = this; }

        /// <summary>
        /// Constructs an instance of the MonitorResourceLock using the specified object as the lock itself.
        /// </summary>
        /// <param name="obj"></param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1720:IdentifiersShouldNotContainTypeNames", MessageId = "obj")]
        public MonitorResourceLock(Object obj)
            : base(ResourceLockOptions.AcquiringThreadMustRelease | ResourceLockOptions.IsMutualExclusive | ResourceLockOptions.SupportsRecursion) { m_lock = obj; }

        /// <summary>
        /// Implements the ResourceLock's WaitToWrite behavior.
        /// </summary>
        protected override void OnEnter(Boolean exclusive)
        {
            Monitor.Enter(m_lock);
        }

        /// <summary>
        /// Implements the ResourceLock's DoneWriting behavior.
        /// </summary>
        protected override void OnLeave(Boolean exclusive)
        {
            Monitor.Exit(m_lock);
        }
    }

    public sealed class OneManyResourceLock : ResourceLock
    {
        #region Lock State Management
#if false
      private struct BitField {
         private Int32 m_mask, m_1, m_startBit;
         public BitField(Int32 startBit, Int32 numBits) {
            m_startBit = startBit;
            m_mask = unchecked((Int32)((1 << numBits) - 1) << startBit);
            m_1 = unchecked((Int32)1 << startBit);
         }
         public void Increment(ref Int32 value) { value += m_1; }
         public void Decrement(ref Int32 value) { value -= m_1; }
         public void Decrement(ref Int32 value, Int32 amount) { value -= m_1 * amount; }
         public Int32 Get(Int32 value) { return (value & m_mask) >> m_startBit; }
         public Int32 Set(Int32 value, Int32 fieldValue) { return (value & ~m_mask) | (fieldValue << m_startBit); }
      }

      private static BitField s_state = new BitField(0, 3);
      private static BitField s_readersReading = new BitField(3, 9);
      private static BitField s_readersWaiting = new BitField(12, 9);
      private static BitField s_writersWaiting = new BitField(21, 9);
      private static OneManyLockStates State(Int32 value) { return (OneManyLockStates)s_state.Get(value); }
      private static void State(ref Int32 ls, OneManyLockStates newState) {
         ls = s_state.Set(ls, (Int32)newState);
      }
#endif
        private enum OneManyLockStates
        {
            Free = 0x00000000,
            OwnedByWriter = 0x00000001,
            OwnedByReaders = 0x00000002,
            OwnedByReadersAndWriterPending = 0x00000003,
            ReservedForWriter = 0x00000004,
        }

        private const Int32 c_lsStateStartBit = 0;
        private const Int32 c_lsReadersReadingStartBit = 3;
        private const Int32 c_lsReadersWaitingStartBit = 12;
        private const Int32 c_lsWritersWaitingStartBit = 21;

        // Mask = unchecked((Int32) ((1 << numBits) - 1) << startBit);
        private const Int32 c_lsStateMask = unchecked((Int32)((1 << 3) - 1) << c_lsStateStartBit);
        private const Int32 c_lsReadersReadingMask = unchecked((Int32)((1 << 9) - 1) << c_lsReadersReadingStartBit);
        private const Int32 c_lsReadersWaitingMask = unchecked((Int32)((1 << 9) - 1) << c_lsReadersWaitingStartBit);
        private const Int32 c_lsWritersWaitingMask = unchecked((Int32)((1 << 9) - 1) << c_lsWritersWaitingStartBit);
        private const Int32 c_lsAnyWaitingMask = c_lsReadersWaitingMask | c_lsWritersWaitingMask;

        // FirstBit = unchecked((Int32) 1 << startBit);
        private const Int32 c_ls1ReaderReading = unchecked((Int32)1 << c_lsReadersReadingStartBit);
        private const Int32 c_ls1ReaderWaiting = unchecked((Int32)1 << c_lsReadersWaitingStartBit);
        private const Int32 c_ls1WriterWaiting = unchecked((Int32)1 << c_lsWritersWaitingStartBit);

        private static OneManyLockStates State(Int32 ls) { return (OneManyLockStates)(ls & c_lsStateMask); }
        private static void SetState(ref Int32 ls, OneManyLockStates newState)
        {
            ls = (ls & ~c_lsStateMask) | ((Int32)newState);
        }

        private static Int32 NumReadersReading(Int32 ls) { return (ls & c_lsReadersReadingMask) >> c_lsReadersReadingStartBit; }
        private static void AddReadersReading(ref Int32 ls, Int32 amount) { ls += (c_ls1ReaderReading * amount); }

        private static Int32 NumReadersWaiting(Int32 ls) { return (ls & c_lsReadersWaitingMask) >> c_lsReadersWaitingStartBit; }
        private static void AddReadersWaiting(ref Int32 ls, Int32 amount) { ls += (c_ls1ReaderWaiting * amount); }

        private static Int32 NumWritersWaiting(Int32 ls) { return (ls & c_lsWritersWaitingMask) >> c_lsWritersWaitingStartBit; }
        private static void AddWritersWaiting(ref Int32 ls, Int32 amount) { ls += (c_ls1WriterWaiting * amount); }

        private static Boolean AnyWaiters(Int32 ls) { return (ls & c_lsAnyWaitingMask) != 0; }

        private static String DebugState(Int32 ls)
        {
            return String.Format(CultureInfo.InvariantCulture,
               "State={0}, RR={1}, RW={2}, WW={3}", State(ls),
               NumReadersReading(ls), NumReadersWaiting(ls),
               NumWritersWaiting(ls));
        }

        /// <summary>
        /// Returns a string representing the state of the object.
        /// </summary>
        /// <returns>The string representing the state of the object.</returns>
        public override String ToString() { return DebugState(m_LockState); }
        #endregion

        #region State Fields
        private Int32 m_LockState = (Int32)OneManyLockStates.Free;

        // Readers wait on this if a writer owns the lock
        private Semaphore m_ReadersLock = new Semaphore(0, Int32.MaxValue);

        // Writers wait on this if a reader owns the lock
        private Semaphore m_WritersLock = new Semaphore(0, Int32.MaxValue);
        #endregion

        #region Construction and Dispose
        /// <summary>Constructs a OneManyLock object.</summary>
        public OneManyResourceLock() : base(ResourceLockOptions.None) { }

        ///<summary>Releases all resources used by the lock.</summary>
        protected override void Dispose(Boolean disposing)
        {
            m_WritersLock.Close(); m_WritersLock = null;
            m_ReadersLock.Close(); m_ReadersLock = null;
            base.Dispose(disposing);
        }
        #endregion

        #region Writer members
        /// <summary>Acquires the lock.</summary>
        protected override void OnEnter(Boolean exclusive)
        {
            if (exclusive)
            {
                while (WaitToWrite(ref m_LockState)) m_WritersLock.WaitOne();
            }
            else
            {
                while (WaitToRead(ref m_LockState)) m_ReadersLock.WaitOne();
            }
        }

        private static Boolean WaitToWrite(ref Int32 target)
        {
            Int32 start, current = target;
            Boolean wait;
            do
            {
                start = current;
                Int32 desired = start;
                wait = false;

                switch (State(desired))
                {
                    case OneManyLockStates.Free:  // If Free -> OBW, return
                    case OneManyLockStates.ReservedForWriter: // If RFW -> OBW, return
                        SetState(ref desired, OneManyLockStates.OwnedByWriter);
                        break;

                    case OneManyLockStates.OwnedByWriter:  // If OBW -> WW++, wait & loop around
                        AddWritersWaiting(ref desired, 1);
                        wait = true;
                        break;

                    case OneManyLockStates.OwnedByReaders: // If OBR or OBRAWP -> OBRAWP, WW++, wait, loop around
                    case OneManyLockStates.OwnedByReadersAndWriterPending:
                        SetState(ref desired, OneManyLockStates.OwnedByReadersAndWriterPending);
                        AddWritersWaiting(ref desired, 1);
                        wait = true;
                        break;
                    default:
                        Debug.Assert(false, "Invalid Lock state");
                        break;
                }
                current = Interlocked.CompareExchange(ref target, desired, start);
            } while (start != current);
            return wait;
        }

        /// <summary>Releases the lock.</summary>
        protected override void OnLeave(Boolean write)
        {
            Int32 wakeup;
            if (write)
            {
                Debug.Assert((State(m_LockState) == OneManyLockStates.OwnedByWriter) && (NumReadersReading(m_LockState) == 0));
                // Pre-condition:  Lock's state must be OBW (not Free/OBR/OBRAWP/RFW)
                // Post-condition: Lock's state must become Free or RFW (the lock is never passed)

                // Phase 1: Release the lock
                wakeup = DoneWriting(ref m_LockState);
            }
            else
            {
                var s = State(m_LockState);
                Debug.Assert((State(m_LockState) == OneManyLockStates.OwnedByReaders) || (State(m_LockState) == OneManyLockStates.OwnedByReadersAndWriterPending));
                // Pre-condition:  Lock's state must be OBR/OBRAWP (not Free/OBW/RFW)
                // Post-condition: Lock's state must become unchanged, Free or RFW (the lock is never passed)

                // Phase 1: Release the lock
                wakeup = DoneReading(ref m_LockState);
            }

            // Phase 2: Possibly wake waiters
            if (wakeup == -1) m_WritersLock.Release();
            else if (wakeup > 0) m_ReadersLock.Release(wakeup);
        }

        // Returns -1 to wake a writer, +# to wake # readers, or 0 to wake no one
        private static Int32 DoneWriting(ref Int32 target)
        {
            Int32 start, current = target;
            Int32 wakeup = 0;
            do
            {
                Int32 desired = (start = current);

                // We do this test first because it is commonly true & 
                // we avoid the other tests improving performance
                if (!AnyWaiters(desired))
                {
                    SetState(ref desired, OneManyLockStates.Free);
                    wakeup = 0;
                }
                else if (NumWritersWaiting(desired) > 0)
                {
                    SetState(ref desired, OneManyLockStates.ReservedForWriter);
                    AddWritersWaiting(ref desired, -1);
                    wakeup = -1;
                }
                else
                {
                    wakeup = NumReadersWaiting(desired);
                    Debug.Assert(wakeup > 0);
                    SetState(ref desired, OneManyLockStates.OwnedByReaders);
                    AddReadersWaiting(ref desired, -wakeup);
                    // RW=0, RR=0 (incremented as readers enter)
                }
                current = Interlocked.CompareExchange(ref target, desired, start);
            } while (start != current);
            return wakeup;
        }
        #endregion

        #region Reader members
        private static Boolean WaitToRead(ref Int32 target)
        {
            Int32 start, current = target;
            Boolean wait;
            do
            {
                Int32 desired = (start = current);
                wait = false;

                switch (State(desired))
                {
                    case OneManyLockStates.Free:  // If Free->OBR, RR=1, return
                        SetState(ref desired, OneManyLockStates.OwnedByReaders);
                        AddReadersReading(ref desired, 1);
                        break;

                    case OneManyLockStates.OwnedByReaders: // If OBR -> RR++, return
                        AddReadersReading(ref desired, 1);
                        break;

                    case OneManyLockStates.OwnedByWriter:  // If OBW/OBRAWP/RFW -> RW++, wait, loop around
                    case OneManyLockStates.OwnedByReadersAndWriterPending:
                    case OneManyLockStates.ReservedForWriter:
                        AddReadersWaiting(ref desired, 1);
                        wait = true;
                        break;

                    default:
                        Debug.Assert(false, "Invalid Lock state");
                        break;
                }
                current = Interlocked.CompareExchange(ref target, desired, start);
            } while (start != current);
            return wait;
        }

        // Returns -1 to wake a writer, +# to wake # readers, or 0 to wake no one
        private static Int32 DoneReading(ref Int32 target)
        {
            Int32 start, current = target;
            Int32 wakeup;
            do
            {
                Int32 desired = (start = current);
                AddReadersReading(ref desired, -1);  // RR--
                if (NumReadersReading(desired) > 0)
                {
                    // RR>0, no state change & no threads to wake
                    wakeup = 0;
                }
                else if (!AnyWaiters(desired))
                {
                    SetState(ref desired, OneManyLockStates.Free);
                    wakeup = 0;
                }
                else
                {
                    Debug.Assert(NumWritersWaiting(desired) > 0);
                    SetState(ref desired, OneManyLockStates.ReservedForWriter);
                    AddWritersWaiting(ref desired, -1);
                    wakeup = -1;   // Wake 1 writer
                }
                current = Interlocked.CompareExchange(ref target, desired, start);
            } while (start != current);
            return wakeup;
        }
        #endregion
    }

    public abstract partial class ResourceLock : IFormattable, IDisposable
    {
#if DEADLOCK_DETECTION
      private static Boolean s_PerformDeadlockDetection = false;

      /// <summary>Turns deadlock detection or or off.</summary>
      /// <param name="enable">true to turn on deadlock detection; false to turn it off.</param>
      [Obsolete("NOTE: Deadlock detection contains a bug that occasionally causes it to report deadlock when deadlock does not actually exist.")]
      public static void PerformDeadlockDetection(Boolean enable) { s_PerformDeadlockDetection = enable; }

      /// <summary>Indicates if deadlock detection is currently on or off.</summary>
      public static Boolean IsDeadlockDetectionOn { get { return s_PerformDeadlockDetection; } }

      ///<summary>Indicates whether deadlock detection applies to this lock or not.</summary>
      ///<returns>True if deadlock detection doesn't apply to this lock.</returns>
      public Boolean ImmuneFromDeadlockDetection {
         get { return (m_resourceLockOptions & ResourceLockOptions.ImmuneFromDeadlockDetection) != 0; }
         set {
            if (value) m_resourceLockOptions |= ResourceLockOptions.ImmuneFromDeadlockDetection;
            else m_resourceLockOptions &= ~ResourceLockOptions.ImmuneFromDeadlockDetection;
         }
      }
#endif

        private String m_name;
        private ResourceLockOptions m_resourceLockOptions;

        /// <summary>Initializes a new instance of a reader/writer lock indicating whether the lock is really a mutual-exclusive lock 
        /// and whether the lock requires that any thread that enters it must be the same thread to exit it.</summary>
        /// <param name="resourceLockOptions">true if this lock really only allows one thread at a time into it; otherwise false.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Performance", "CA1805:DoNotInitializeUnnecessarily")]
        protected ResourceLock(ResourceLockOptions resourceLockOptions)
        {
            m_resourceLockOptions = resourceLockOptions;
            InitConditionalVariableSupport();
        }

        partial void InitConditionalVariableSupport();

        ///<summary>Releases all resources used by the reader/writer lock.</summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ///<summary>Releases all resources used by the lock.</summary>
        [SuppressMessage("Microsoft.Usage", "CA2213:DisposableFieldsShouldBeDisposed",
           Justification = "m_doneWritingDisposer and m_doneReadingDisposer don't represent native resources")]
        protected virtual void Dispose(Boolean disposing)
        {
        }

        /// <summary>Returns options that describe the behavior of this lock.</summary>
        public ResourceLockOptions ResourceLockOptions { get { return m_resourceLockOptions; } }

        // High 16 bits is num writers in lock; low 16-bits is num readers
        private const Int32 c_OneReaderCount = 0x0001;
        private const Int32 c_OneWriterCount = 0x10000;
        private Int32 m_readWriteCounts = 0;

        /// <summary>Returns the number of reader threads currently owning the lock.</summary>
        /// <returns>The number of reader threads in the lock.</returns>
        public Int32 CurrentReaderCount() { return m_readWriteCounts & 0xffff; }

        /// <summary>Returns the number of writer threads currently owning the lock.</summary>
        /// <returns>The number of writer threads in the lock.</returns>
        public Int32 CurrentWriterCount() { return m_readWriteCounts >> 16; }

        /// <summary>Returns true if no thread currently owns the lock.</summary>
        /// <returns>true if no thread currently own the lock.</returns>
        public Boolean CurrentlyFree() { return m_readWriteCounts == 0; }

        ///<summary>Indicates whether the lock treats all requests as mutual-exclusive.</summary>
        ///<returns>True if the lock class allows just one thread at a time.</returns>
        public Boolean IsMutualExclusive
        {
            get { return (m_resourceLockOptions & ResourceLockOptions.IsMutualExclusive) != 0; }
        }

        ///<summary>Indicates whether the lock supports recursion.</summary>
        ///<returns>True if the lock supports recursion.</returns>
        public Boolean SupportsRecursion
        {
            get { return (m_resourceLockOptions & ResourceLockOptions.SupportsRecursion) != 0; }
        }

        ///<summary>Indicates whether the thread that acquires the lock must also release the lock.</summary>
        ///<returns>True if the thread that requires the lock must also release it.</returns>
        public Boolean AcquiringThreadMustRelease
        {
            get { return (m_resourceLockOptions & ResourceLockOptions.AcquiringThreadMustRelease) != 0; }
        }

        /// <summary>
        /// The name associated with this lock for debugging purposes.
        /// </summary>
        public String Name
        {
            get { return m_name; }
            set
            {
                if (m_name == null) m_name = value;
                else throw new InvalidOperationException("This property has already been set and cannot be modified.");
            }
        }

        // NOTE: All locks must implement the WaitToWrite/DoneWriting methods
        ///<summary>Allows the calling thread to acquire the lock for writing or reading.</summary>
        /// <param name="exclusive">true if the thread wishes to acquire the lock for exclusive access.</param>
        public void Enter(Boolean exclusive)
        {
#if DEADLOCK_DETECTION
         if (exclusive) {
            if (AcquiringThreadMustRelease) Thread.BeginCriticalRegion();
            using (s_PerformDeadlockDetection ? DeadlockDetector.BlockForLock(this, true) : null) {
               OnEnter(exclusive);
            }
         } else {
            // When reading, there is no need to call BeginCriticalRegion since resource is not being modified
            using (s_PerformDeadlockDetection ? DeadlockDetector.BlockForLock(this, IsMutualExclusive) : null) {
               OnEnter(exclusive);
            }
         }
#else
            OnEnter(exclusive);
#endif
            Interlocked.Add(ref m_readWriteCounts, exclusive ? c_OneWriterCount : c_OneReaderCount);
        }

        ///<summary>Derived class overrides <c>OnEnter</c> to provide specific lock-acquire semantics.</summary>
        protected abstract void OnEnter(Boolean exclusive);

        ///<summary>Derived class overrides <c>OnLeave</c> to provide specific lock-release semantics.</summary>
        protected abstract void OnLeave(Boolean exclusive);

        ///<summary>Allows the calling thread to release the lock.</summary>
        public void Leave()
        {
            Contract.Assume(!CurrentlyFree());
            Boolean exclusive = CurrentReaderCount() == 0;
#if DEADLOCK_DETECTION
         if (s_PerformDeadlockDetection) DeadlockDetector.ReleaseLock(this);
#endif
            OnLeave(exclusive);
            if (exclusive)
            {
                Interlocked.Add(ref m_readWriteCounts, -c_OneWriterCount);
                //if (AcquiringThreadMustRelease) Thread.EndCriticalRegion();
            }
            else
            {
                Interlocked.Add(ref m_readWriteCounts, -c_OneReaderCount);
                // When done reading, there is no need to call EndCriticalRegion since resource was not modified
            }
        }

        #region Helper Methods

        ///<summary>If<c>Stress</c> is defined during compilation, calls to this method cause the calling thread to sleep.</summary>
        [Conditional("Stress")]
        protected static void StressPause() { Thread.Sleep(2); }

        ///<summary>Allows calling thread to yield CPU time to another thread.</summary>
        protected static void StallThread() { ThreadUtility.StallThread(); }

        ///<summary>Compares two values for equality and, if they are equal, replaces one of the values.</summary>
        ///<return>Returns true if the value in <paramref name="value"/> was equal the the value of <paramref name="if"/>.</return>
        ///<param name="value">The destination, whose value is compared with <paramref name="if"/> and possibly replaced with <paramref name="then"/>.</param>
        ///<param name="if">The value that is compared to the value at <paramref name="value"/>.</param>
        ///<param name="then">The value that might get placed into <paramref name="value"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        protected static Boolean IfThen(ref Int32 value, Int32 @if, Int32 then)
        {
            return InterlockedEx.IfThen(ref value, @if, then);
        }

        ///<summary>Compares two values for equality and, if they are equal, replaces one of the values.</summary>
        ///<remarks>The previous value in <paramref name="value"/> is returned in <paramref name="previousValue"/>.</remarks>
        ///<return>Returns true if the value in <paramref name="value"/> was equal the the value of <paramref name="if"/>.</return>
        ///<param name="value">The destination, whose value is compared with <paramref name="if"/> and possibly replaced with <paramref name="then"/>.</param>
        ///<param name="if">The value that is compared to the value at <paramref name="value"/>.</param>
        ///<param name="then">The value that might get placed into <paramref name="value"/>.</param>
        ///<param name="previousValue">The previous value that was in <paramref name="value"/> prior to calling this method.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#"), SuppressMessage("Microsoft.Design", "CA1021:AvoidOutParameters")]
        protected static Boolean IfThen(ref Int32 value, Int32 @if, Int32 then, out Int32 previousValue)
        {
            return InterlockedEx.IfThen(ref value, @if, then, out previousValue);
        }

        #endregion

        #region IFormattable Members
        ///<summary>Returns the object's string representation.</summary>
        ///<return>A <c>String</c> containing the object's string representation.</return>
        public String ToString(String format) { return ToString(format, null); }

        ///<summary>Returns the object's string representation.</summary>
        ///<return>A <c>String</c> containing the object's string representation.</return>
        public String ToString(IFormatProvider formatProvider)
        {
            return ToString(null, formatProvider);
        }

        ///<summary>Returns the object's string representation.</summary>
        ///<param name="format">If <c>null</c> or <c>"extra"</c> is allowed.</param>
        ///<param name="formatProvider">Not used.</param>
        ///<return>A <c>String</c> containing the object's string representation.</return>
        public virtual String ToString(String format, IFormatProvider formatProvider)
        {
            if (format == null) return ToString();
            if (String.Compare(format, "extra", StringComparison.OrdinalIgnoreCase) == 0)
                return ToString();
            throw new FormatException("Unknown format string: " + format);
        }

        /// <summary>
        /// Returns a System.String that represents the current System.Object.
        /// </summary>
        /// <returns>Returns a System.String that represents the current System.Object.</returns>
        public override string ToString()
        {
            return m_name ?? base.ToString();
        }

        #endregion

        /// <summary>
        /// Determines whether the specified System.Object is equal to the current System.Object.
        /// </summary>
        /// <param name="obj">The System.Object to compare with the current System.Object.</param>
        /// <returns>true if the specified System.Object is equal to the current System.Object; otherwise, false.</returns>
        public override Boolean Equals(Object obj)
        {
            ResourceLock other = obj as ResourceLock;
            if (other == null) return false;
            return (GetType() == other.GetType()) && (m_resourceLockOptions == other.m_resourceLockOptions);
        }

        /// <summary>
        /// Serves as a hash function for a particular type.
        /// </summary>
        /// <returns>A hash code for the current System.Object.</returns>
        public override Int32 GetHashCode()
        {
            return base.GetHashCode();
        }

#if DEBUG
        public void Hammer()
        {
            for (Int32 n = 0; n < 5; n++)
            {
                Hammer(10, 100);
                // The lock should settle back to Free here
            }
            System.Console.WriteLine("done");
            System.Console.ReadLine();
        }

        private volatile Boolean m_stop = false;
        private readonly Random m_rand = new Random();

        private void Hammer(Int32 exclusive, Int32 shared)
        {
            System.Console.WriteLine("Hammering {0} exclusive & {1} shared starting.", exclusive, shared);
            m_stop = false;
            Int32 threads = exclusive + shared;
            List<Thread> l = new List<Thread>();
            Int32 writersInLock = 0, readersInLock = 0;

            for (Int32 n = 0; n < threads; n++)
            {
                Thread t = new Thread(num =>
                {
                    Boolean exclusiveThread = ((Int32)num) < exclusive;
                    while (!m_stop)
                    {
                        Enter(exclusiveThread);
                        if (exclusiveThread)
                        {
                            if (Interlocked.Increment(ref writersInLock) != 1 || Thread.VolatileRead(ref readersInLock) != 0) Debugger.Break();
                        }
                        else
                        {
                            if (Interlocked.Increment(ref readersInLock) > shared || Thread.VolatileRead(ref writersInLock) != 0) Debugger.Break();
                        }
                        System.Console.WriteLine("   ThreadNum={0,3}, Writers={1}, Readers={2}", num, writersInLock, readersInLock);

                        // Body
                        var bodyWork = m_rand.Next(10) * (exclusiveThread ? 100 : 10);
                        for (var end = Environment.TickCount + bodyWork; Environment.TickCount < end; ) ;

                        if (exclusiveThread) Interlocked.Decrement(ref writersInLock);
                        else Interlocked.Decrement(ref readersInLock);
                        Leave();

                        var iterationSleep = m_rand.Next(100) * (exclusiveThread ? 100 : 10);
                        Thread.Sleep(iterationSleep);
                    }
                });
                t.Name = n.ToString() + " " + ((n < exclusive) ? " exclusive" : " shared");
                l.Add(t);
                t.Start(n);
            }
            Thread.Sleep(TimeSpan.FromMinutes(5));
            m_stop = true;
            foreach (var t in l) t.Join();
            System.Console.WriteLine("Hammering {0} exclusive & {1} shared completed.", exclusive, shared);
        }
#endif
    }

    public class AsyncResult : IAsyncResult
    {
        // Fields set at construction which never change while operation is pending
        private readonly AsyncCallback m_AsyncCallback;
        private readonly Object m_AsyncState;
        private readonly Object m_InitiatingObject = null;

        // Field set at construction which do change after operation completes
        private const Int32 c_StatePending = 0;
        private const Int32 c_StateCompletedSynchronously = 1;
        private const Int32 c_StateCompletedAsynchronously = 2;
        private Int32 m_CompletedState = c_StatePending;

        // Field that may or may not get set depending on usage
        private volatile ManualResetEvent m_AsyncWaitHandle;
        private Int32 m_eventSet = 0; // 0=false, 1= true

        // Fields set when operation completes
        private Exception m_exception;

        // Find method to retain Exception's stack trace when caught and rethrown
        // NOTE: GetMethod returns null if method is not available
        private static readonly MethodInfo s_Exception_InternalPreserveStackTrace =
           typeof(Exception).GetMethod("InternalPreserveStackTrace", BindingFlags.Instance | BindingFlags.NonPublic);

        /// <summary>
        /// Constructs an object that identifies an asynchronous operation.
        /// </summary>
        /// <param name="asyncCallback">The method that should be executed when the operation completes.</param>
        /// <param name="state">The object that can be obtained via the AsyncState property.</param>
        public AsyncResult(AsyncCallback asyncCallback, Object state)
        {
            m_AsyncCallback = asyncCallback;
            m_AsyncState = state;
        }

        /// <summary>
        /// Constructs an object that identifies an asynchronous operation.
        /// </summary>
        /// <param name="asyncCallback">The method that should be executed when the operation completes.</param>
        /// <param name="state">The object that can be obtained via the AsyncState property.</param>
        /// <param name="initiatingObject">Identifies the object that is initiating the asynchronous operation. This object is obtainable via the InitiatingObject property.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1720:IdentifiersShouldNotContainTypeNames", MessageId = "object")]
        public AsyncResult(AsyncCallback asyncCallback, Object state, Object initiatingObject)
            : this(asyncCallback, state)
        {
            m_InitiatingObject = initiatingObject;
        }

        /// <summary>
        /// Gets the object passed to the constructor to initiate the asynchronous operation.
        /// </summary>
        public Object InitiatingObject { get { return m_InitiatingObject; } }

#if !SILVERLIGHT && !PocketPC
        private static Exception PreserveExceptionStackTrace(Exception exception)
        {
            if (exception == null) return null;

            // Try the fast/hacky way first: Call Exception's non-public InternalPreserveStackTrace method to do it
            if (s_Exception_InternalPreserveStackTrace != null)
            {
                try
                {
                    s_Exception_InternalPreserveStackTrace.Invoke(exception, null);
                    return exception;
                }
                catch (MethodAccessException)
                {
                    // Method can't be accessed, try serializing/deserializing the exception
                }
            }

            // The hacky way failed: Serialize and deserialize the exception object
            using (MemoryStream ms = new MemoryStream(1000))
            {
                // Using CrossAppDomain causes the Exception to retain its stack
                BinaryFormatter formatter = new BinaryFormatter(null, new StreamingContext(StreamingContextStates.CrossAppDomain));
                formatter.Serialize(ms, exception);

                ms.Seek(0, SeekOrigin.Begin);
                return (Exception)formatter.Deserialize(ms);
            }
        }
#endif

        /// <summary>
        /// Call this method to indicate that the asynchronous operation has completed.
        /// </summary>
        /// <param name="exception">If non-null, this argument identifies the exception that occurring while processing the asynchronous operation.</param>
        /// <param name="completedSynchronously">Indicates whether the operation completed synchronously or asynchronously.</param>
        public void SetAsCompleted(Exception exception, Boolean completedSynchronously)
        {
            // Passing null for exception means no error occurred; this is the common case
#if !SILVERLIGHT && !PocketPC
            m_exception = PreserveExceptionStackTrace(exception);
#else
         m_exception = exception;
#endif

            // The m_CompletedState field MUST be set prior to calling the callback
            Int32 prevState = Interlocked.Exchange(ref m_CompletedState,
               completedSynchronously ? c_StateCompletedSynchronously : c_StateCompletedAsynchronously);
            if (prevState != c_StatePending)
                throw new InvalidOperationException("You can set a result only once");

            // If the event exists and it hasn't been set yet, set it
            ManualResetEvent mre = m_AsyncWaitHandle; // This is a volatile read
            if ((mre != null) && CallingThreadShouldSetTheEvent())
                mre.Set();

            // If a callback method was set, call it
            if (m_AsyncCallback != null) m_AsyncCallback(this);
        }

        /// <summary>
        /// Frees up resources used by the asynchronous operation represented by the IAsyncResult passed.
        /// If the asynchronous operation failed, this method throws the exception.
        /// </summary>
        public void EndInvoke()
        {
            // This method assumes that only 1 thread calls EndInvoke for this object

            // If the operation isn't done or if the wait handle was created, wait for it
            if (!IsCompleted || (m_AsyncWaitHandle != null))
                AsyncWaitHandle.WaitOne();

            // If the wait handle was created, close it
#pragma warning disable 420
            ManualResetEvent mre = Interlocked.Exchange(ref m_AsyncWaitHandle, null);
#pragma warning restore 420
            if (mre != null) mre.Close();

            // Operation is done: if an exception occurred, throw it
            if (m_exception != null) throw m_exception;
        }

        #region Implementation of IAsyncResult
        /// <summary>
        /// Gets a user-defined object that qualifies or contains information about an asynchronous operation.
        /// </summary>
        public Object AsyncState { get { return m_AsyncState; } }

        /// <summary>
        /// Gets an indication of whether the asynchronous operation completed synchronously.
        /// </summary>
        public Boolean CompletedSynchronously
        {
            get
            {
#if PocketPC || SILVERLIGHT   // No Thread.Volatile methods
            Thread.MemoryBarrier();
            return m_CompletedState == c_StateCompletedSynchronously; 
#else
                return Thread.VolatileRead(ref m_CompletedState) == c_StateCompletedSynchronously;
#endif
            }
        }

        private Boolean CallingThreadShouldSetTheEvent() { return (Interlocked.Exchange(ref m_eventSet, 1) == 0); }

        /// <summary>
        /// Gets a WaitHandle that is used to wait for an asynchronous operation to complete.
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Reliability", "CA2000:Dispose objects before losing scope")]
        public WaitHandle AsyncWaitHandle
        {
            get
            {
                Contract.Ensures(Contract.Result<WaitHandle>() != null);
                if (m_AsyncWaitHandle == null)
                {
                    ManualResetEvent mre = new ManualResetEvent(false);
#pragma warning disable 420
                    if (Interlocked.CompareExchange(ref m_AsyncWaitHandle, mre, null) != null)
                    {
#pragma warning restore 420
                        Contract.Assume(m_AsyncWaitHandle != null);  // Remove when I.CE gets a post-condition contract
                        // Another thread created this object's event; dispose the event we just created
                        mre.Close();
                    }
                    else
                    {
                        // This thread created the event. 
                        // If the operation is complete and no other thread set the event, then this thread should set it
                        if (IsCompleted && CallingThreadShouldSetTheEvent())
                        {
                            //Contract.Assume(m_AsyncWaitHandle != null);  // Remove when I.CE gets a post-condition contract
                            m_AsyncWaitHandle.Set();
                        }
                    }
                }
                Contract.Assume(m_AsyncWaitHandle != null);  // Remove when I.CE gets a post-condition contract
                return m_AsyncWaitHandle;
            }
        }

        /// <summary>
        /// Gets an indication whether the asynchronous operation has completed.
        /// </summary>
        public Boolean IsCompleted
        {
            get
            {
#if PocketPC || SILVERLIGHT  // No Thread.Volatile methods
            Thread.MemoryBarrier();
            return m_CompletedState != c_StatePending; 
#else
                return Thread.VolatileRead(ref m_CompletedState) != c_StatePending;
#endif
            }
        }
        #endregion


        #region Helper Members
        private static readonly AsyncCallback s_AsyncCallbackHelper = AsyncCallbackCompleteOpHelperNoReturnValue;

        /// <summary>
        /// Returns a single static delegate to a static method that will invoke the desired AsyncCallback
        /// </summary>
        /// <returns>The single static delegate.</returns>
        protected static AsyncCallback GetAsyncCallbackHelper() { return s_AsyncCallbackHelper; }

        private static WaitCallback s_WaitCallbackHelper = WaitCallbackCompleteOpHelperNoReturnValue;

        /// <summary>
        /// Returns an IAsyncResult for an operations that was queued to the thread pool.
        /// </summary>
        /// <returns>The IAsyncResult.</returns>
        protected IAsyncResult BeginInvokeOnWorkerThread()
        {
            ThreadPool.QueueUserWorkItem(s_WaitCallbackHelper, this);
            return this;
        }

        // This static method allows us to have just one static delegate 
        // instead of constructing a delegate per instance of this class
        private static void AsyncCallbackCompleteOpHelperNoReturnValue(IAsyncResult otherAsyncResult)
        {
            Contract.Requires(otherAsyncResult != null);
            AsyncResult ar = (AsyncResult)otherAsyncResult.AsyncState;
            Contract.Assume(ar != null);
            ar.CompleteOpHelper(otherAsyncResult);
        }

        private static void WaitCallbackCompleteOpHelperNoReturnValue(Object o)
        {
            Contract.Requires(o != null);
            AsyncResult ar = (AsyncResult)o;
            ar.CompleteOpHelper(null);
        }

        [SuppressMessage("Microsoft.Design", "CA1031:DoNotCatchGeneralExceptionTypes",
           Justification = "OK because exception will be thrown by EndInvoke.")]
        private void CompleteOpHelper(IAsyncResult ar)
        {
            Exception exception = null;
            try
            {
                OnCompleteOperation(ar);
            }
            catch (TargetInvocationException e)
            {
                exception = e.InnerException;
            }
            catch (Exception e)
            {
                exception = e;
            }
            finally
            {
                SetAsCompleted(exception, false);
            }
        }

        /// <summary>
        /// Invokes the callback method when the asynchronous operations completes.
        /// </summary>
        /// <param name="result">The IAsyncResult object identifying the asynchronous operation that has completed.</param>
        protected virtual void OnCompleteOperation(IAsyncResult result) { }
        #endregion
    }

    public class AsyncResult<TResult> : AsyncResult
    {
        // Field set when operation completes
        private TResult m_result;

        /// <summary>
        /// Constructs an object that identifies an asynchronous operation.
        /// </summary>
        /// <param name="asyncCallback">The method that should be executed wehen the operation completes.</param>
        /// <param name="state">The object that can be obtained via the AsyncState property.</param>
        public AsyncResult(AsyncCallback asyncCallback, Object state)
            : base(asyncCallback, state)
        {
        }

        /// <summary>
        /// Constructs an object that identifies an asynchronous operation.
        /// </summary>
        /// <param name="asyncCallback">The method that should be executed wehen the operation completes.</param>
        /// <param name="state">The object that can be obtained via the AsyncState property.</param>
        /// <param name="initiatingObject">Identifies the object that is initiating the asynchronous operation. This object is obtainable via the InitiatingObject property.</param>
        public AsyncResult(AsyncCallback asyncCallback, Object state, Object initiatingObject)
            : base(asyncCallback, state, initiatingObject)
        {
        }

        /// <summary>
        /// Call this method to indicate that the asynchronous operation has completed.
        /// </summary>
        /// <param name="result">Indicates the value calculated by the asynchronous operation.</param>
        /// <param name="completedSynchronously">Indicates whether the operation completed synchronously or asynchronously.</param>
        public void SetAsCompleted(TResult result, Boolean completedSynchronously)
        {
            m_result = result;
            base.SetAsCompleted(null, completedSynchronously);
        }

        /// <summary>
        /// Frees up resources used by the asynchronous operation represented by the IAsyncResult passed.
        /// If the asynchronous operation failed, this method throws the exception. If the operation suceeded,
        /// this method returns the value calculated by the asynchronous operation.
        /// </summary>
        /// <returns>The value calculated by the asynchronous operation.</returns>
        public new TResult EndInvoke()
        {
            base.EndInvoke(); // Wait until operation has completed 
            return m_result;  // Return the result (if above didn't throw)
        }

        #region Helper Members
        private static readonly AsyncCallback s_AsyncCallbackHelper = AsyncCallbackCompleteOpHelperWithReturnValue;

        /// <summary>
        /// Returns a single static delegate to a static method that will invoke the desired AsyncCallback
        /// </summary>
        /// <returns>The single static delegate.</returns>
        [SuppressMessage("Microsoft.Design", "CA1000:DoNotDeclareStaticMembersOnGenericTypes",
           Justification = "OK since member is protected")]
        protected new static AsyncCallback GetAsyncCallbackHelper() { return s_AsyncCallbackHelper; }

        private static void AsyncCallbackCompleteOpHelperWithReturnValue(IAsyncResult otherAsyncResult)
        {
            Contract.Requires(otherAsyncResult != null);
            Contract.Requires(otherAsyncResult.AsyncState != null);
            AsyncResult<TResult> ar = (AsyncResult<TResult>)otherAsyncResult.AsyncState;
            ar.CompleteOpHelper(otherAsyncResult);
        }

        private static WaitCallback s_WaitCallbackHelper = WaitCallbackCompleteOpHelperWithReturnValue;

        /// <summary>
        /// Returns an IAsyncResult for an operations that was queued to the thread pool.
        /// </summary>
        /// <returns>The IAsyncResult.</returns>
        protected new IAsyncResult BeginInvokeOnWorkerThread()
        {
            ThreadPool.QueueUserWorkItem(s_WaitCallbackHelper, this);
            return this;
        }
        private static void WaitCallbackCompleteOpHelperWithReturnValue(Object o)
        {
            Contract.Requires(o != null);
            AsyncResult<TResult> ar = (AsyncResult<TResult>)o;
            ar.CompleteOpHelper(null);
        }

        [SuppressMessage("Microsoft.Design", "CA1031:DoNotCatchGeneralExceptionTypes",
           Justification = "OK because exception will be thrown by EndInvoke.")]
        private void CompleteOpHelper(IAsyncResult ar)
        {
            TResult result = default(TResult);
            Exception exception = null;
            try
            {
                result = OnCompleteOperation(ar);
            }
            catch (Exception e)
            {
                exception = (e is TargetInvocationException) ? e.InnerException : e;
            }
            if (exception == null) SetAsCompleted(result, false);
            else SetAsCompleted(exception, false);
        }

        /// <summary>
        /// Invokes the callback method when the asynchronous operations completes.
        /// </summary>
        /// <param name="result">The IAsyncResult object identifying the asynchronous operation that has completed.</param>
        /// <returns>The value computed by the asynchronous operation.</returns>
        protected new virtual TResult OnCompleteOperation(IAsyncResult result)
        {
            return default(TResult);
        }
        #endregion
    }

    public static class InterlockedEx
    {
        #region Generic Morph and Morpher
        /// <summary>Identifies a method that morphs the Int32 startValue into a new value, returning it.</summary>
        /// <typeparam name="TResult">The return type returned by the Morph method.</typeparam>
        /// <typeparam name="TArgument">The argument type passed to the Morph method.</typeparam>
        /// <param name="startValue">The initial Int32 value.</param>
        /// <param name="argument">The argument passed to the method.</param>
        /// <param name="morphResult">The value returned from Morph when the morpher callback method is successful.</param>
        /// <returns>The value that the morpher method desires to set the <paramref name="startValue"/> to.</returns>
        [SuppressMessage("Microsoft.Design", "CA1034:NestedTypesShouldNotBeVisible"), System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1704:IdentifiersShouldBeSpelledCorrectly", MessageId = "Morpher")]
        public delegate Int32 Morpher<TResult, TArgument>(Int32 startValue, TArgument argument, out TResult morphResult);

        /// <summary>Atomically modifies an Int32 value using an algorithm identified by <paramref name="morpher"/>.</summary>
        /// <typeparam name="TResult">The type of the return value.</typeparam>
        /// <typeparam name="TArgument">The type of the argument passed to the <paramref name="morpher"/> callback method.</typeparam>
        /// <param name="target">A reference to the Int32 value that is to be modified atomically.</param>
        /// <param name="argument">A value of type <typeparamref name="TArgument"/> that will be passed on to the <paramref name="morpher"/> callback method.</param>
        /// <param name="morpher">The algorithm that modifies the Int32 returning a new Int32 value and another return value to be returned to the caller.</param>
        /// <returns>The desired Int32 value.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1704:IdentifiersShouldBeSpelledCorrectly", MessageId = "morpher"), System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static TResult Morph<TResult, TArgument>(ref Int32 target, TArgument argument, Morpher<TResult, TArgument> morpher)
        {
            Contract.Requires(morpher != null);
            TResult morphResult;
            Int32 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, morpher(i, argument, out morphResult), i);
            } while (i != j);
            return morphResult;
        }
        #endregion

        #region Convenience Wrappers
        ///<summary>Compares two values for equality and, if they are equal, replaces one of the values.</summary>
        ///<return>Returns true if the value in <paramref name="value"/> was equal the the value of <paramref name="if"/>.</return>
        ///<param name="value">The destination, whose value is compared with <paramref name="if"/> and possibly replaced with <paramref name="then"/>.</param>
        ///<param name="if">The value that is compared to the value at <paramref name="value"/>.</param>
        ///<param name="then">The value that might get placed into <paramref name="value"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Boolean IfThen(ref Int32 value, Int32 @if, Int32 then)
        {
            return (Interlocked.CompareExchange(ref value, then, @if) == @if);
        }

#if !SILVERLIGHT && !PocketPC
        ///<summary>Compares two values for equality and, if they are equal, replaces one of the values.</summary>
        ///<return>Returns true if the value in <paramref name="value"/> was equal the the value of <paramref name="if"/>.</return>
        ///<param name="value">The destination, whose value is compared with <paramref name="if"/> and possibly replaced with <paramref name="then"/>.</param>
        ///<param name="if">The value that is compared to the value at <paramref name="value"/>.</param>
        ///<param name="then">The value that might get placed into <paramref name="value"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Boolean IfThen(ref UInt32 value, UInt32 @if, UInt32 then)
        {
            return (InterlockedEx.CompareExchange(ref value, then, @if) == @if);
        }
#endif

        ///<summary>Compares two values for equality and, if they are equal, replaces one of the values.</summary>
        ///<remarks>The previous value in <paramref name="value"/> is returned in <paramref name="previousValue"/>.</remarks>
        ///<return>Returns true if the value in <paramref name="value"/> was equal the the value of <paramref name="if"/>.</return>
        ///<param name="value">The destination, whose value is compared with <paramref name="if"/> and possibly replaced with <paramref name="then"/>.</param>
        ///<param name="if">The value that is compared to the value at <paramref name="value"/>.</param>
        ///<param name="then">The value that might get placed into <paramref name="value"/>.</param>
        ///<param name="previousValue">The previous value that was in <paramref name="value"/> prior to calling this method.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#"), SuppressMessage("Microsoft.Design", "CA1021:AvoidOutParameters")]
        public static Boolean IfThen(ref Int32 value, Int32 @if, Int32 then, out Int32 previousValue)
        {
            previousValue = Interlocked.CompareExchange(ref value, then, @if);
            return (previousValue == @if);
        }

#if !SILVERLIGHT && !PocketPC
        ///<summary>Compares two values for equality and, if they are equal, replaces one of the values.</summary>
        ///<remarks>The previous value in <paramref name="value"/> is returned in <paramref name="previousValue"/>.</remarks>
        ///<return>Returns true if the value in <paramref name="value"/> was equal the the value of <paramref name="if"/>.</return>
        ///<param name="value">The destination, whose value is compared with <paramref name="if"/> and possibly replaced with <paramref name="then"/>.</param>
        ///<param name="if">The value that is compared to the value at <paramref name="value"/>.</param>
        ///<param name="then">The value that might get placed into <paramref name="value"/>.</param>
        ///<param name="previousValue">The previous value that was in <paramref name="value"/> prior to calling this method.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#"), SuppressMessage("Microsoft.Design", "CA1021:AvoidOutParameters")]
        public static Boolean IfThen(ref UInt32 value, UInt32 @if, UInt32 then, out UInt32 previousValue)
        {
            previousValue = InterlockedEx.CompareExchange(ref value, then, @if);
            return (previousValue == @if);
        }
#endif

        ///<summary>Compares two values for equality and, if they are equal, replaces one of the values.</summary>
        ///<return>Returns true if the value in <paramref name="value"/> was equal the the value of <paramref name="if"/>.</return>
        ///<typeparam name="T">The type to be used for value, if, and then. This type must be a reference type.</typeparam>
        ///<param name="value">The destination, whose value is compared with <paramref name="if"/> and possibly replaced with <paramref name="then"/>.</param>
        ///<param name="if">The value that is compared to the value at <paramref name="value"/>.</param>
        ///<param name="then">The value that might get placed into <paramref name="value"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Boolean IfThen<T>(ref T value, T @if, T then) where T : class
        {
            return (Interlocked.CompareExchange(ref value, then, @if) == @if);
        }

        ///<summary>Compares two values for equality and, if they are equal, replaces one of the values.</summary>
        ///<remarks>The previous value in <paramref name="value"/> is returned in <paramref name="previousValue"/>.</remarks>
        ///<return>Returns true if the value in <paramref name="value"/> was equal the the value of <paramref name="if"/>.</return>
        ///<typeparam name="T">The type to be used for value, if, and then. This type must be a reference type.</typeparam>
        ///<param name="value">The destination, whose value is compared with <paramref name="if"/> and possibly replaced with <paramref name="then"/>.</param>
        ///<param name="if">The value that is compared to the value at <paramref name="value"/>.</param>
        ///<param name="then">The value that might get placed into <paramref name="value"/>.</param>
        ///<param name="previousValue">The previous value that was in <paramref name="value"/> prior to calling this method.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#"), SuppressMessage("Microsoft.Design", "CA1021:AvoidOutParameters")]
        public static Boolean IfThen<T>(ref T value, T @if, T then, out T previousValue) where T : class
        {
            previousValue = Interlocked.CompareExchange(ref value, then, @if);
            return (previousValue == @if);
        }

#if !SILVERLIGHT && !PocketPC
        /// <summary>Compares two 32-bit unsigned integers for equality and, if they are equal, replaces one of the values.</summary>
        /// <remarks>If comparand and the value in location are equal, then value is stored in location. Otherwise, no operation is performed. The compare and exchange operations are performed as an atomic operation. The return value of CompareExchange is the original value in location, whether or not the exchange takes place.</remarks>
        /// <param name="location">The destination, whose value is compared with comparand and possibly replaced.</param>
        /// <param name="value">The value that replaces the destination value if the comparison results in equality.</param>
        /// <param name="comparand">The value that is compared to the value at <paramref name="location"/>.</param>
        /// <returns>The original value in location.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static UInt32 CompareExchange(ref UInt32 location, UInt32 value, UInt32 comparand)
        {
            unsafe
            {
                fixed (UInt32* p = &location)
                {
                    Int32* p2 = (Int32*)p;
                    return (UInt32)Interlocked.CompareExchange(ref *p2, (Int32)value, (Int32)comparand);
                }
            }
        }
#endif

#if !SILVERLIGHT && !PocketPC
        /// <summary>Sets a 32-bit unsigned integer to a specified value and returns the original value, as an atomic operation.</summary>
        /// <param name="location">The variable to set to the specified value.</param>
        /// <param name="value">The value to which the <paramref name="location"/> parameter is set.</param>
        /// <returns>The original value of <paramref name="location"/>.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static UInt32 Exchange(ref UInt32 location, UInt32 value)
        {
            unsafe
            {
                fixed (UInt32* p = &location)
                {
                    Int32* p2 = (Int32*)p;
                    return (UInt32)Interlocked.Exchange(ref *p2, (Int32)value);
                }
            }
        }
#endif
        #endregion

        #region Mathematic Operations
#if !SILVERLIGHT && !PocketPC
        /// <summary>Adds a 32-bit signed integer to a 32-bit unsigned integer and replaces the first integer with the sum, as an atomic operation.</summary>
        /// <param name="location">A variable containing the first value to be added. The sum of the two values is stored in <paramref name="location"/>.</param>
        /// <param name="value">The value to be added to the integer at <paramref name="location"/>.</param>
        /// <returns>The new value stored at <paramref name="location"/>.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static UInt32 Add(ref UInt32 location, Int32 value)
        {
            unsafe
            {
                fixed (UInt32* p = &location)
                {
                    Int32* p2 = (Int32*)p;
                    return (UInt32)Interlocked.Add(ref *p2, value);
                }
            }
        }
#endif

#if PocketPC
      /// <summary>Adds a 32-bit signed integer to a 32-bit signed integer and replaces the first integer with the sum, as an atomic operation.</summary>
      /// <param name="target">A variable containing the first value to be added. The sum of the two values is stored in <paramref name="target"/>.</param>
      /// <param name="value">The value to be added to the integer at <paramref name="target"/>.</param>
      /// <returns>The new value stored at <paramref name="target"/>.</returns>
      [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
      public static Int32 Add(ref Int32 target, Int32 value) {
         Int32 i, j = target;
         do {
            i = j;
            j = Interlocked.CompareExchange(ref target, i + value, i);
         } while (i != j);
         return j;
      }
#endif

#if !SILVERLIGHT && !PocketPC
        /// <summary>Increments a specified variable and stores the result, as an atomic operation.</summary>
        /// <param name="location">The variable whose value is to be incremented.</param>
        /// <returns>The incremented value.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static UInt32 Increment(ref UInt32 location) { return Add(ref location, 1); }


        /// <summary>Decrements a specified variable and stores the result, as an atomic operation.</summary>
        /// <param name="location">The variable whose value is to be decremented.</param>
        /// <returns>The decremented value.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static UInt32 Decrement(ref UInt32 location) { return Add(ref location, -1); }
#endif

        ///<summary>Increases a value to a new value if the new value is larger.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the value that might be increased to a new maximum.</param>
        ///<param name="value">The value that if larger than <paramref name="target"/> will be placed in <paramref name="target"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 Max(ref Int32 target, Int32 value)
        {
            Int32 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, Math.Max(i, value), i);
            } while (i != j);
            return j;
        }

        ///<summary>Increases a value to a new value if the new value is larger.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the value that might be increased to a new maximum.</param>
        ///<param name="value">The value that if larger than <paramref name="target"/> will be placed in <paramref name="target"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int64 Max(ref Int64 target, Int64 value)
        {
            Int64 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, Math.Max(i, value), i);
            } while (i != j);
            return j;
        }

        ///<summary>Decreases a value to a new value if the new value is smaller.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the value that might be decreased to a new minimum.</param>
        ///<param name="value">The value that if smaller than <paramref name="target"/> will be placed in <paramref name="target"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 Min(ref Int32 target, Int32 value)
        {
            Int32 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, Math.Min(i, value), i);
            } while (i != j);
            return j;
        }

        ///<summary>Decreases a value to a new value if the new value is smaller.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the value that might be decreased to a new minimum.</param>
        ///<param name="value">The value that if smaller than <paramref name="target"/> will be placed in <paramref name="target"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int64 Min(ref Int64 target, Int64 value)
        {
            Int64 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, Math.Min(i, value), i);
            } while (i != j);
            return j;
        }

        ///<summary>Decrements a value by 1 if the value is greater than the specified value (usually 0).</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the value that might be decremented.</param>
        ///<param name="lowValue">The value that target must be greater than in order for the decrement to occur.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 DecrementIfGreaterThan(ref Int32 target, Int32 lowValue)
        {
            Int32 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, (i > lowValue) ? (i - 1) : i, i);
            } while (i != j);
            return j;
        }


        /// <summary>Decrements a value by 1 if the value is greater than 0.</summary>
        /// <param name="target">A variable containing the value that might be decremented.</param>
        /// <param name="value">The value to add to target before calculating the modulo specified in <paramref name="modulo"/>.</param>
        /// <param name="modulo">The value to use for the modulo operation.</param>
        /// <returns>Returns the previous value of <paramref name="target"/>.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 AddModulo(ref Int32 target, Int32 value, Int32 modulo)
        {
            Contract.Requires(modulo != 0);
            Int32 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, (i + value) % modulo, i);
            } while (i != j);
            return j;
        }
        #endregion

        #region Bit Operations
        ///<summary>Bitwise ANDs two 32-bit integers and replaces the first integer with the ANDed value, as an atomic operation.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the first value to be ANDed. The bitwise AND of the two values is stored in <paramref name="target"/>.</param>
        ///<param name="with">The value to AND with <paramref name="target"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 And(ref Int32 target, Int32 with)
        {
            Int32 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, i & with, i);
            } while (i != j);
            return j;
        }

        ///<summary>Bitwise ORs two 32-bit integers and replaces the first integer with the ORed value, as an atomic operation.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the first value to be ORed. The bitwise OR of the two values is stored in <paramref name="target"/>.</param>
        ///<param name="with">The value to OR with <paramref name="target"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 Or(ref Int32 target, Int32 with)
        {
            Int32 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, i | with, i);
            } while (i != j);
            return j;
        }

#if !PocketPC
        ///<summary>Bitwise ORs two 64-bit signed integers and replaces the first integer with the ORed value, as an atomic operation.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the first value to be ORed. The bitwise OR of the two values is stored in <paramref name="target"/>.</param>
        ///<param name="with">The value to OR with <paramref name="target"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int64 Or(ref Int64 target, Int64 with)
        {
            Int64 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, i | with, i);
            } while (i != j);
            return j;
        }

        ///<summary>Bitwise XORs two 32-bit integers and replaces the first integer with the XORed value, as an atomic operation.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the first value to be XORed. The bitwise XOR of the two values is stored in <paramref name="target"/>.</param>
        ///<param name="with">The value to XOR with <paramref name="target"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 Xor(ref Int32 target, Int32 with)
        {
            Int32 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, i ^ with, i);
            } while (i != j);
            return j;
        }

        ///<summary>Turns a bit on and returns whether or not it was on.</summary>
        ///<return>Returns whether the bit was on prior to calling this method.</return>
        ///<param name="target">A variable containing the value that is to have a bit turned on.</param>
        ///<param name="bitNumber">The bit (0-31) in <paramref name="target"/> that should be turned on.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Boolean BitTestAndSet(ref Int32 target, Int32 bitNumber)
        {
            Int32 tBit = unchecked((Int32)(1u << bitNumber));
            // Turn the bit on and return if it was on
            return (Or(ref target, tBit) & tBit) != 0;
        }

        ///<summary>Turns a bit off and returns whether or not it was on.</summary>
        ///<return>Returns whether the bit was on prior to calling this method.</return>
        ///<param name="target">A variable containing the value that is to have a bit turned off.</param>
        ///<param name="bitNumber">The bit (0-31) in <paramref name="target"/> that should be turned off.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Boolean BitTestAndReset(ref Int32 target, Int32 bitNumber)
        {
            Int32 tBit = unchecked((Int32)(1u << bitNumber));
            // Turn the bit off and return if it was on
            return (And(ref target, ~tBit) & tBit) != 0;
        }

        ///<summary>Flips an on bit off or and off bit on.</summary>
        ///<return>Returns whether the bit was on prior to calling this method.</return>
        ///<param name="target">A variable containing the value that is to have a bit flipped.</param>
        ///<param name="bitNumber">The bit (0-31) in <paramref name="target"/> that should be flipped.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Boolean BitTestAndCompliment(ref Int32 target, Int32 bitNumber)
        {
            Int32 tBit = unchecked((Int32)(1u << bitNumber));
            // Toggle the bit and return if it was on
            return (Xor(ref target, tBit) & tBit) != 0;
        }
#endif
        #endregion

        #region Masked Bit Operations
        ///<summary>Bitwise ANDs two 32-bit integers with a mask replacing the first integer with the ANDed value, as an atomic operation.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the first value to be ANDed. The bitwise AND of the two values is stored in <paramref name="target"/>.</param>
        ///<param name="with">The value to AND with <paramref name="target"/>.</param>
        ///<param name="mask">The value to AND with <paramref name="target"/> prior to ANDing with <paramref name="with"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 MaskedAnd(ref Int32 target, Int32 with, Int32 mask)
        {
            Int32 i, j = target;
            do
            {
                i = j & mask;  // Mask off the bits we're not interested in
                j = Interlocked.CompareExchange(ref target, i & with, i);
            } while (i != j);
            return j;
        }

        ///<summary>Bitwise ORs two 32-bit integers with a mask replacing the first integer with the ORed value, as an atomic operation.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the first value to be ORed. The bitwise OR of the two values is stored in <paramref name="target"/>.</param>
        ///<param name="with">The value to OR with <paramref name="target"/>.</param>
        ///<param name="mask">The value to AND with <paramref name="target"/> prior to ORing with <paramref name="with"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 MaskedOr(ref Int32 target, Int32 with, Int32 mask)
        {
            Int32 i, j = target;
            do
            {
                i = j & mask;  // Mask off the bits we're not interested in
                j = Interlocked.CompareExchange(ref target, i | with, i);
            } while (i != j);
            return j;
        }

        ///<summary>Bitwise XORs two 32-bit integers with a mask replacing the first integer with the XORed value, as an atomic operation.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the first value to be XORed. The bitwise XOR of the two values is stored in <paramref name="target"/>.</param>
        ///<param name="with">The value to XOR with <paramref name="target"/>.</param>
        ///<param name="mask">The value to AND with <paramref name="target"/> prior to XORing with <paramref name="with"/>.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 MaskedXor(ref Int32 target, Int32 with, Int32 mask)
        {
            Int32 i, j = target;
            do
            {
                i = j & mask;  // Mask off the bits we're not interested in
                j = Interlocked.CompareExchange(ref target, i ^ with, i);
            } while (i != j);
            return j;
        }

        ///<summary>Sets a variable to a specified value as an atomic operation.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the value to be replaced.</param>
        ///<param name="mask">The bits to leave unaffected in <paramref name="target"/> prior to ORing with <paramref name="value"/>.</param>
        ///<param name="value">The value to replace <paramref name="target"/> with.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 MaskedExchange(ref Int32 target, Int32 mask, Int32 value)
        {
            Int32 i, j = target;
            do
            {
                i = j;
                j = Interlocked.CompareExchange(ref target, (i & ~mask) | value, j);
            } while (i != j);
            return j;
        }

        ///<summary>Adds two integers and replaces the first integer with the sum, as an atomic operation.</summary>
        ///<return>Returns the previous value of <paramref name="target"/>.</return>
        ///<param name="target">A variable containing the value to be replaced.</param>
        ///<param name="value">The value to add to <paramref name="target"/>.</param>
        ///<param name="mask">The bits in <paramref name="target"/> that should not be affected by adding.</param>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1045:DoNotPassTypesByReference", MessageId = "0#")]
        public static Int32 MaskedAdd(ref Int32 target, Int32 value, Int32 mask)
        {
            Int32 i, j = target;
            do
            {
                i = j & mask;  // Mask off the bits we're not interested in
                j = Interlocked.CompareExchange(ref target, i + value, i);
            } while (i != j);
            return j;
        }
        #endregion
    }
       public static class ThreadUtility {
      #region Set Name of Finalizer Thread
      /// <summary>
      /// This method sets the name of the Finalizer thread for viewing in the debugger
      /// </summary>
      /// <param name="name">The string to name the Finalizer thread.</param>
      [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA1806:DoNotIgnoreMethodResults", MessageId = "Wintellect.Threading.ThreadUtility+SetNameOfFinalizerThread"), System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Reliability", "CA2001:AvoidCallingProblematicMethods", MessageId = "System.GC.Collect")]
      public static void NameFinalizerThreadForDebugging(String name) {
         Contract.Requires(name != null);
         new SetNameOfFinalizerThread(name);
         GC.Collect();
         GC.WaitForPendingFinalizers();
      }
      private class SetNameOfFinalizerThread {
         private String m_name;
         public SetNameOfFinalizerThread() : this("Finalizer thread") { }
         public SetNameOfFinalizerThread(String name) {
            Contract.Requires(name != null);  m_name = name;
         }
         ~SetNameOfFinalizerThread() { Thread.CurrentThread.Name = m_name; }
         [ContractInvariantMethod]
         void ObjectInvariant() {
            Contract.Invariant(m_name != null);
         }
      }
      #endregion

      /// <summary>
      /// Returns true if the host machine has just one CPU.
      /// </summary>
      public static readonly Boolean IsSingleCpuMachine =
         (Environment.ProcessorCount == 1);

      /// <summary>
      /// Blocks the calling thread for the specified time.
      /// </summary>
      /// <param name="milliseconds">The number of milliseconds that this method should wait before returning.</param>
      /// <param name="computeBound">true if this method should spin in a compute bound loop; false if 
      /// Windows should not schedule for the specified amount of time.</param>
      public static void Block(Int32 milliseconds, Boolean computeBound) {
         if (computeBound) {
            Int64 stop = milliseconds + Environment.TickCount;
            while (Environment.TickCount < stop) ;
         } else { Thread.Sleep(milliseconds); }
      }

      /// <summary>
      /// Returns a ProcessThread object for a specified Win32 thread Id.
      /// </summary>
      /// <param name="threadId">The Win32 thread Id value.</param>
      /// <returns>A ProcessThread object matching the specified thread Id.</returns>
      [ContractVerification(false)]
      public static ProcessThread GetProcessThreadFromWin32ThreadId(Int32 threadId) {
         if (threadId == 0) threadId = ThreadUtility.GetCurrentWin32ThreadId();
         foreach (Process process in Process.GetProcesses()) {
            foreach (ProcessThread processThread in process.Threads) {
               if (processThread.Id == threadId) return processThread;
            }
         }
         throw new InvalidOperationException("No thread matching specified thread Id was found.");
      }

      #region Simple Win32 Thread Wrappers
      /// <summary>
      /// Returns the Win32 thread Id matching the thread that created the specified window handle.
      /// </summary>
      /// <param name="hwnd">Identifies a window handle.</param>
      /// <returns>The thread that created the window.</returns>
      //public static Int32 GetWindowThreadId(HWND hwnd) {
      //   Int32 processId;
      //   return NativeMethods.GetWindowThreadProcessId(hwnd, out processId);
      //}

      ///// <summary>
      ///// Returns the Win32 process Id containing the thread that created the specified window handle.
      ///// </summary>
      ///// <param name="hwnd">Identifies a window handle.</param>
      ///// <returns>The process owning the thread that created the window.</returns>
      //[System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1704:IdentifiersShouldBeSpelledCorrectly", MessageId = "hwnd")]
      //public static Int32 GetWindowProcessId(HWND hwnd) {
      //   Int32 processId;
      //   Int32 threadId = NativeMethods.GetWindowThreadProcessId(hwnd, out processId);
      //   return processId;
      //}

      ///// <summary>
      ///// Opens a thread in the system identified via its Win32 thread Id.
      ///// </summary>
      ///// <param name="rights">Indicates how you intend to manipulate the thread.</param>
      ///// <param name="inheritHandle">true if the returned handle should be inherited by child processes.</param>
      ///// <param name="threadId">The Win32 Id identifying a thread.</param>
      ///// <returns>A SafeWaitHandle matching the opened thread. This method throws a WaitHandleCannotBeOpenedException if the thread cannot be opened.</returns>
      //public static SafeWaitHandle OpenThread(ThreadRights rights, Boolean inheritHandle, Int32 threadId) {
      //   SafeWaitHandle thread = NativeMethods.OpenThread(rights, inheritHandle, threadId);
      //   Contract.Assume(thread != null);
      //   if (thread.IsInvalid) throw new WaitHandleCannotBeOpenedException();
      //   return thread;
      //}

      /// <summary>
      /// Retrieves the number of the processor the current thread was running on during the call to this function.
      /// </summary>
      /// <returns>The current processor number.</returns>
      [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1024:UsePropertiesWhereAppropriate")]
      public static Int32 GetCurrentProcessorNumber() { return NativeMethods.GetCurrentProcessorNumber(); }


      /// <summary>
      /// Retrieves the Win32 Id of the calling thread.
      /// </summary>
      /// <returns>The Win32 thread Id of the calling thread.</returns>
      [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1024:UsePropertiesWhereAppropriate")]
      public static Int32 GetCurrentWin32ThreadId() { return NativeMethods.GetCurrentWin32ThreadId(); }

      /// <summary>
      /// Retrieves a pseudo handle for the calling thread.
      /// </summary>
      /// <returns>The pseudo handle for the current thread.</returns>
      [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1024:UsePropertiesWhereAppropriate")]
      public static SafeWaitHandle GetCurrentWin32ThreadHandle() { return NativeMethods.GetCurrentWin32ThreadHandle(); }

      /// <summary>
      /// Retrieves a pseudo handle for the calling thread's process.
      /// </summary>
      /// <returns>The pseudo handle for the current process.</returns>
      [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1024:UsePropertiesWhereAppropriate")]
      public static SafeWaitHandle GetCurrentWin32ProcessHandle() { return NativeMethods.GetCurrentWin32ProcessHandle(); }

      /// <summary>
      /// Causes the calling thread to yield execution to another thread 
      /// that is ready to run on the current processor. The operating 
      /// system selects the next thread to be executed.
      /// </summary>
      /// <returns>true if the operating system switches execution to another thread; 
      /// false if there are no other threads ready to execute and the OS doesn't switch 
      /// execution to another thread.</returns>
      public static Boolean SwitchToThread() { return NativeMethods.SwitchToThread(); }

      /// <summary>
      /// Tells the I/O Manager to not signal the file/device 
      /// handle when an I/O operation completes.
      /// </summary>
      /// <param name="device">Identifies the file or device that should not be signaled.</param>
      public static void SkipSignalOfDeviceOnIOCompletion(SafeFileHandle device) {
         if (!IsVistaOrLaterOS()) return;
         if (!NativeMethods.SetFileCompletionNotificationModes(device, FileCompletionNotificationNodes.FILE_SKIP_SET_EVENT_ON_HANDLE))
            throw new Win32Exception();
      }

      /// <summary>
      /// Tells the I/O Manager to not queue a completion entry to the specified 
      /// device's I/O completion port if the I/O operation completes synchronously.
      /// </summary>
      /// <param name="device">Identifies the file or device whose 
      /// synchronously-executed operation should not be placed in an 
      /// I/O completion port.</param>
      public static void SkipCompletionPortOnSynchronousIOCompletion(SafeFileHandle device) {
         ValidateVistaOrLaterOS();
         if (!NativeMethods.SetFileCompletionNotificationModes(device, FileCompletionNotificationNodes.FILE_SKIP_COMPLETION_PORT_ON_SUCCESS))
            throw new Win32Exception();
      }

      private enum FileCompletionNotificationNodes : byte {
         FILE_SKIP_COMPLETION_PORT_ON_SUCCESS = 1,
         FILE_SKIP_SET_EVENT_ON_HANDLE = 2
      }

      /// <summary>
      /// Causes the calling thread to allow another thread to run.
      /// </summary>
      public static void StallThread() {
         if (IsSingleCpuMachine) {
            // On single-CPU system, spinning does no good
            SwitchToThread();
         } else {
            // The multi-CPU system might be hyper-threaded, let the other thread run
            Thread.SpinWait(1);
         }
      }

      /// <summary>
      /// Retrieves the cycle time for the specified thread.
      /// </summary>
      /// <param name="threadHandle">Identifies the thread whose cycle time you'd like to obtain.</param>
      /// <returns>The thread's cycle time.</returns>
      [CLSCompliant(false)]
      public static UInt64 QueryThreadCycleTime(SafeWaitHandle threadHandle) {
         ValidateVistaOrLaterOS();
         UInt64 cycleTime;
         if (!NativeMethods.QueryThreadCycleTime(threadHandle, out cycleTime))
            throw new Win32Exception();
         return cycleTime;
      }

      /// <summary>
      /// Retrieves the sum of the cycle time of all threads of the specified process.
      /// </summary>
      /// <param name="processHandle">Identifies the process whose threads' cycles times you'd like to obtain.</param>
      /// <returns>The process' cycle time.</returns>
      [CLSCompliant(false)]
      public static UInt64 QueryProcessCycleTime(SafeWaitHandle processHandle) {
         ValidateVistaOrLaterOS();
         UInt64 cycleTime;
         if (!NativeMethods.QueryProcessCycleTime(processHandle, out cycleTime))
            throw new Win32Exception();
         return cycleTime;
      }

      /// <summary>
      /// Retrieves the cycle time for the idle thread of each processor in the system.
      /// </summary>
      /// <returns>The number of CPU clock cycles used by each idle thread.</returns>
      [CLSCompliant(false)]
      public static UInt64[] QueryIdleProcessorCycleTimes() {
         ValidateVistaOrLaterOS();
         Int32 byteCount = Environment.ProcessorCount;
         Contract.Assume(byteCount > 0);
         UInt64[] cycleTimes = new UInt64[byteCount];
         byteCount *= 8;   // Size of UInt64
         if (!NativeMethods.QueryIdleProcessorCycleTime(ref byteCount, cycleTimes))
            throw new Win32Exception();
         return cycleTimes;
      }
      #endregion

      #region Cancel Synchronous I/O
      /*
		Not cancellable: 
			DeleteTree (use WalkTree), CopyFile (use CopyFileEx), MoveFile(Ex) (use MoveFileWithProgress), ReplaceFile

		Cancellable via callback: 
			WalkTree, CopyFileEx, MoveFileWithProgress

		Cancellable via CancelSynchronousIo: 
			CreateFile, ReadFile(Ex), ReadFileScatter, WriteFile(Ex), WriteFileGather, SetFilePointer(Ex),
			SetEndOfFile, SetFileValidData, FlushFileBuffers, LockFile(Ex), UnlockFile(Ex), 
			FindClose, FindFirstFile(Ex), FindNextFile, FindFirstStreamW, FindNextStreamW,
			CreateHardLink, DeleteFile, GetFileType, GetBinaryType, 
			GetCompressedFileSize, GetFileInformationByHandle, GetFileAttributes(Ex), SetFileAttributes, 
			GetFileSize(Ex), GetFileTime, SetFileTime, SetFileSecurity
			GetFullPathName, GetLongPathName, GetShortPathName, SetFileShortName, 
			GetTempFileName, GetTempPath, SearchPath, 
			GetQueuedCompletionStatus,
			CreateFileMapping, MapViewOfFile(Ex), FlushViewOfFile			
      */

      /// <summary>
      /// Marks pending synchronous I/O operations that are issued by the specified thread as canceled.
      /// </summary>
      /// <param name="thread">Identifies the thread whose synchronous I/O you want to cancel.</param>
      /// <returns>true if an operation is cancelled; false if the thread was not waiting for I/O</returns>
      public static Boolean CancelSynchronousIO(SafeWaitHandle thread) {
         ValidateVistaOrLaterOS();
         if (NativeMethods.CancelSynchronousIO(thread)) return true;
         Int32 error = Marshal.GetLastWin32Error();

         const Int32 ErrorNotFound = 1168;
         if (error == ErrorNotFound) return false; // failed to cancel because thread was not waiting

         throw new Win32Exception(error);
      }
      #endregion

      #region I/O Background Processing Mode
    //  private static readonly Disposer s_endBackgroundProcessingMode = new Disposer(EndBackgroundProcessingMode);

      /// <summary>
      /// The system lowers the resource scheduling priorities of the thread 
      /// so that it can perform background work without significantly 
      /// affecting activity in the foreground.
      /// </summary>
      /// <returns>An IDisposable object that can be used to end 
      /// background processing mode for the thread.</returns>
      //public static IDisposable BeginBackgroundProcessingMode() {
      //   ValidateVistaOrLaterOS();
      //   if (NativeMethods.SetThreadPriority(GetCurrentWin32ThreadHandle(), BackgroundProcessingMode.Start))
      //      return s_endBackgroundProcessingMode;
      //   throw new Win32Exception();
      //}

      ///// <summary>
      ///// The system restores the resource scheduling priorities of the thread 
      ///// as they were before the thread entered background processing mode.
      ///// </summary>
      //public static void EndBackgroundProcessingMode() {
      //   ValidateVistaOrLaterOS();
      //   if (NativeMethods.SetThreadPriority(GetCurrentWin32ThreadHandle(), BackgroundProcessingMode.End))
      //      return;
      //   throw new Win32Exception();
      //}

      private enum BackgroundProcessingMode {
         Start = 0x10000,
         End = 0x20000
      }
      #endregion

      private static Boolean IsVistaOrLaterOS() {
         OperatingSystem os = Environment.OSVersion;
         Contract.Assume(os != null);
         return (os.Version >= new Version(6, 0));
      }

      private static void ValidateVistaOrLaterOS() {
         if (!IsVistaOrLaterOS())
            throw new NotSupportedException("Requires Windows 6.0 or later");
      }

      private static class NativeMethods {
         //[DllImport("User32")]
         //internal static extern Int32 GetWindowThreadProcessId(HWND hwnd, out Int32 pdwProcessId);

         //[DllImport("Kernel32", SetLastError = true, EntryPoint = "OpenThread")]
         //internal static extern SafeWaitHandle OpenThread(ThreadRights dwDesiredAccess,
         //   [MarshalAs(UnmanagedType.Bool)] Boolean bInheritHandle, Int32 threadId);

         [DllImport("Kernel32", ExactSpelling = true)]
         internal static extern Int32 GetCurrentProcessorNumber();

         [DllImport("Kernel32", EntryPoint = "GetCurrentThreadId", ExactSpelling = true)]
         internal static extern Int32 GetCurrentWin32ThreadId();

         [DllImport("Kernel32", EntryPoint = "GetCurrentThread", ExactSpelling = true)]
         internal static extern SafeWaitHandle GetCurrentWin32ThreadHandle();

         [DllImport("Kernel32", EntryPoint = "GetCurrentProcess", ExactSpelling = true)]
         internal static extern SafeWaitHandle GetCurrentWin32ProcessHandle();

         [DllImport("Kernel32", ExactSpelling = true)]
         [return: MarshalAs(UnmanagedType.Bool)]
         internal static extern Boolean SwitchToThread();

         // http://msdn.microsoft.com/en-us/library/aa480216.aspx
         [DllImport("Kernel32", SetLastError = true, EntryPoint = "CancelSynchronousIo")]
         [return: MarshalAs(UnmanagedType.Bool)]
         internal static extern Boolean CancelSynchronousIO(SafeWaitHandle hThread);

         [DllImport("Kernel32", ExactSpelling = true)]
         [return: MarshalAs(UnmanagedType.Bool)]
         internal static extern Boolean QueryThreadCycleTime(SafeWaitHandle threadHandle, out UInt64 CycleTime);

         [DllImport("Kernel32", ExactSpelling = true)]
         [return: MarshalAs(UnmanagedType.Bool)]
         internal static extern Boolean QueryProcessCycleTime(SafeWaitHandle processHandle, out UInt64 CycleTime);

         [DllImport("Kernel32", ExactSpelling = true)]
         [return: MarshalAs(UnmanagedType.Bool)]
         internal static extern Boolean QueryIdleProcessorCycleTime(ref Int32 byteCount, UInt64[] CycleTimes);

         [DllImport("Kernel32", ExactSpelling = true)]
         [return: MarshalAs(UnmanagedType.Bool)]
         internal static extern Boolean SetFileCompletionNotificationModes(SafeFileHandle FileHandle, FileCompletionNotificationNodes Flags);

         [DllImport("Kernel32", ExactSpelling = true, SetLastError = true)]
         [return: MarshalAs(UnmanagedType.Bool)]
         internal static extern Boolean SetThreadPriority(SafeWaitHandle hthread, BackgroundProcessingMode mode);
      }
   }
}

