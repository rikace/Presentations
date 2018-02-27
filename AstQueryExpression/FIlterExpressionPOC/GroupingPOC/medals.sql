USE [Olympics]
GO

/****** Object: Table [dbo].[Medals] Script Date: 07/10/2016 16:04:07 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[Medals] (
    [Games]      NVARCHAR (50)  NOT NULL,
    [Year]       INT            NOT NULL,
    [Sport]      NVARCHAR (50)  NOT NULL,
    [Discipline] NVARCHAR (50)  NOT NULL,
    [Athlete]    NVARCHAR (100) NOT NULL,
    [Team]       NVARCHAR (100) NOT NULL,
    [Gender]     NVARCHAR (50)  NOT NULL,
    [Event]      NVARCHAR (100) NOT NULL,
    [Metal]      NVARCHAR (50)  NOT NULL,
    [Gold]       INT            NOT NULL,
    [Silver]     INT            NOT NULL,
    [Bronze]     INT            NOT NULL,
    [ID]         INT            IDENTITY (1, 1) NOT NULL
);


