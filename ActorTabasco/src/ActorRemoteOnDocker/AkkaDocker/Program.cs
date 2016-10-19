using System;
using Akka.Actor;

namespace AkkaDocker
{
	public class TestActor : UntypedActor
	{
		protected override void OnReceive(object message)
		{
			if(message is Ping)
				Console.WriteLine("Hello world");
		}
	}

	public class Ping { }


	class MainClass
	{
		public static void Main(string[] args)
		{
			Console.WriteLine("Akka Docker is starting up!");
			var system = ActorSystem.Create("System");
			var actor = system.ActorOf(Props.Create(() => new TestActor()));

			system.Scheduler.ScheduleTellRepeatedly(TimeSpan.Zero, TimeSpan.FromSeconds(10), actor, new Ping(), ActorRefs.NoSender);

			Console.WriteLine("Press Enter to exit.");
			Console.ReadLine();
		}
	}
}
