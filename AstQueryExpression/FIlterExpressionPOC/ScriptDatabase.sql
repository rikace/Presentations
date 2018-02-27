SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

USE [master];
GO

IF EXISTS (SELECT * FROM sys.databases WHERE name = 'SimpleDB')
DROP DATABASE SimpleDB;
GO

-- Create the MyDatabase database.
CREATE DATABASE SimpleDB COLLATE SQL_Latin1_General_CP1_CI_AS;
GO

-- Specify a simple recovery model
-- to keep the log growth to a minimum.
ALTER DATABASE SimpleDB
SET RECOVERY SIMPLE;
GO

USE SimpleDB;
GO

CREATE TABLE [dbo].[Courses] (
[CourseID]   INT           NOT NULL,
[Name] NVARCHAR (50) NOT NULL,
[Hours] int NOT NULL,
PRIMARY KEY CLUSTERED ([CourseID] ASC)
);

CREATE TABLE [dbo].[Students] (
[StudentID] INT           NOT NULL,
[FirstName]      NVARCHAR (50) NOT NULL,
[LastName]      NVARCHAR (50) NOT NULL,
[Name]      NVARCHAR (50) NOT NULL,
[Age]       INT           NOT NULL,
PRIMARY KEY CLUSTERED ([StudentID] ASC)
);

CREATE TABLE [dbo].[CourseSelection] (
[ID]        INT NOT NULL,
[StudentID] INT NOT NULL,
[CourseID]  INT NOT NULL,
PRIMARY KEY CLUSTERED ([ID] ASC),
CONSTRAINT [FK_CourseSelection_ToTable] FOREIGN KEY ([StudentID]) REFERENCES [dbo].[Students] ([StudentID]) ON DELETE NO ACTION ON UPDATE NO ACTION,
CONSTRAINT [FK_CourseSelection_Course_1] FOREIGN KEY ([CourseID]) REFERENCES [dbo].[Courses] ([CourseID]) ON DELETE NO ACTION ON UPDATE NO ACTION
);

CREATE TABLE [dbo].[LastStudent] (
[StudentID] INT           NOT NULL,
[FirstName]      NVARCHAR (50) NOT NULL,
[LastName]      NVARCHAR (50) NOT NULL,
[Age]       INT           NULL,
PRIMARY KEY CLUSTERED ([StudentID] ASC)
);

-- Insert data into the tables.
USE SimpleDB
INSERT INTO Courses (CourseID, Name, Hours)
VALUES(1, 'Algebra I', 5);
INSERT INTO Courses (CourseID, Name, Hours)
VALUES(2, 'Trigonometry', 9);
INSERT INTO Courses (CourseID, Name, Hours)
VALUES(3, 'Algebra II', 10);
INSERT INTO Courses (CourseID, Name, Hours)
VALUES(4, 'History', 10);
INSERT INTO Courses (CourseID, Name, Hours)
VALUES(5, 'English', 42);
INSERT INTO Courses (CourseID, Name, Hours)
VALUES(6, 'French', 1);
INSERT INTO Courses (CourseID, Name, Hours)
VALUES(7, 'Chinese', 1000);

INSERT INTO Students (StudentID, FirstName, LastName,Name, Age)
VALUES(1, 'Abercrombie', 'Kim','Abercrombie, Kim', 10);
INSERT INTO Students (StudentID, FirstName, LastName,Name, Age)
VALUES(2, 'Abolrous', ' Hazen','Abolrous Hazen', 14);
INSERT INTO Students (StudentID, FirstName, LastName,Name,Age)
VALUES(3, 'Hance', ' Jim', 'Hance Jim', 12);
INSERT INTO Students (StudentID, FirstName, LastName,Name,Age)
VALUES(4, 'Adams', ' Terry','Adams Terry', 12);
INSERT INTO Students (StudentID, FirstName, LastName,Name,Age)
VALUES(5, 'Hansen', ' Claus','Hansen Claus', 11);
INSERT INTO Students (StudentID, FirstName, LastName,Name,Age)
VALUES(6, 'Penor', ' Lori', 'Penor Lori',13);
INSERT INTO Students (StudentID, FirstName, LastName,Name,Age)
VALUES(7, 'Perham', ' Tom', 'Perham Tom',12);
INSERT INTO Students (StudentID, FirstName, LastName,Name,Age)
VALUES(8, 'Peng', ' Yun-Feng', 'Pen Yun-Feng', 20);

INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(1, 1, 2);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(2, 1, 3);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(3, 1, 5);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(4, 2, 2);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(5, 2, 5);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(6, 2, 6);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(7, 2, 3);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(8, 3, 2);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(9, 3, 1);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(10, 4, 2);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(11, 4, 5);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(12, 4, 2);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(13, 5, 3);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(14, 5, 2);
INSERT INTO CourseSelection (ID, StudentID, CourseID)
VALUES(15, 7, 3);