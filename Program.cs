using System.Collections.Concurrent;
using System.Diagnostics;
using Spectre.Console;

namespace OsuDan;

public class Program
{
	public static async Task<int> Main(string[] args)
	{
		string? customSongs = null;
		string? customDb = null;
		int concurrency = Environment.ProcessorCount;
		int? limit = null;
		bool dryRun = false;
		bool zeroPadded = true;
		bool purgeDans = false;

		for (int i = 0; i < args.Length; i++)
		{
			switch (args[i].ToLowerInvariant())
			{
				case "-h" or "--help":
					AnsiConsole.MarkupLine("[bold]danorganizer[/]\n");
					AnsiConsole.MarkupLine("[bold]Usage:[/] [cyan]danorganizer [[options]][/]\n");
					AnsiConsole.MarkupLine("[bold]Options:[/]");
					AnsiConsole.MarkupLine("  [cyan]--songs <path>[/]       Path to osu! Songs folder");
					AnsiConsole.MarkupLine("  [cyan]--db <path>[/]          Path to osu! collection.db");
					AnsiConsole.MarkupLine("  [cyan]--threads <n>[/]        Worker thread count (default: CPU cores)");
					AnsiConsole.MarkupLine("  [cyan]--limit <n>[/]          Process at most N maps");
					AnsiConsole.MarkupLine("  [cyan]--test, --dry-run[/]    Test run without writing to collection.db");
					AnsiConsole.MarkupLine("  [cyan]--simple-names[/]       Use simple names ('Dan 1'..'Dan 10', 'Alpha'..'Kappa')");
					AnsiConsole.MarkupLine("  [cyan]--purge-dans[/]         Remove previously generated Dan collections before adding");
					AnsiConsole.MarkupLine("  [cyan]-h, --help[/]           Show this help message\n");
					return 0;
				case "--songs" when i + 1 < args.Length:
					customSongs = args[++i];
					break;
				case "--db" when i + 1 < args.Length:
					customDb = args[++i];
					break;
				case "--threads" when i + 1 < args.Length:
					if (int.TryParse(args[++i], out var t)) concurrency = t;
					break;
				case "--limit" when i + 1 < args.Length:
					if (int.TryParse(args[++i], out var l)) limit = l;
					break;
				case "--dry-run" or "--test":
					dryRun = true;
					break;
				case "--simple-names":
					zeroPadded = false;
					break;
				case "--purge-dans":
					purgeDans = true;
					break;
			}
		}

		var songsDir = customSongs ?? DetectSongsDirectory();
		var dbPath = customDb ?? DetectCollectionDb(songsDir);

		var tableConfig = new Table()
			.Border(TableBorder.Rounded)
			.AddColumn("[bold]Configuration[/]")
			.AddColumn("[bold]Value[/]");

		tableConfig.AddRow("Songs", songsDir ?? "[red]Not found[/]");
		tableConfig.AddRow("collection.db", dbPath ?? "[red]Not found[/]");
		tableConfig.AddRow("Mode", dryRun ? "[yellow]Test[/]" : "[green]Live[/]");
		if (limit.HasValue) tableConfig.AddRow("Limit", limit.Value.ToString());

		AnsiConsole.Write(tableConfig);
		AnsiConsole.WriteLine();

		if (string.IsNullOrEmpty(songsDir) || !Directory.Exists(songsDir))
		{
			AnsiConsole.MarkupLine("[red]Error:[/] osu! Songs folder not found. Specify with --songs <path>");
			return 1;
		}

		if (string.IsNullOrEmpty(dbPath) && !dryRun)
		{
			AnsiConsole.MarkupLine("[red]Error:[/] collection.db not found. Specify with --db <path>");
			return 1;
		}

		MsdCalculator msdCalculator;
		DanClassifier classifier;

		try
		{
			msdCalculator = new MsdCalculator();
			classifier = new DanClassifier();
		}
		catch (Exception ex)
		{
			AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
			return 1;
		}

		using (classifier)
		{
			var processor = new BatchProcessor(msdCalculator, classifier, zeroPadded: zeroPadded);
			var sw = Stopwatch.StartNew();

			AnsiConsole.MarkupLine("[bold]Scanning beatmaps...[/]\n");

			await AnsiConsole.Progress()
				.AutoRefresh(true)
				.AutoClear(false)
				.HideCompleted(false)
				.Columns(new ProgressColumn[]
				{
					new TaskDescriptionColumn(),
					new ProgressBarColumn(),
					new PercentageColumn(),
					new RemainingTimeColumn(),
					new SpinnerColumn()
				})
				.StartAsync(async ctx =>
				{
					var task = ctx.AddTask("[green]Processing Beatmaps[/]", true, 100);

					var progressHandler = new Progress<BatchProgress>(p =>
					{
						if (task.MaxValue != p.TotalFiles)
							task.MaxValue = p.TotalFiles;

						task.Value = p.ProcessedFiles;
						task.Description = p.CachedHits > 0
							? $"[green]Processed:[/] {p.ProcessedFiles}/{p.TotalFiles} | [grey]Cached:[/] {p.CachedHits}"
							: $"[green]Processed:[/] {p.ProcessedFiles}/{p.TotalFiles}";
					});

					await processor.ProcessSongsAsync(songsDir, concurrency, progressHandler, limit);
				});

			sw.Stop();
			AnsiConsole.WriteLine();
			AnsiConsole.MarkupLine($"[bold green]Completed in {sw.Elapsed.TotalSeconds:F1}s[/]\n");

			DisplaySummaryTable(processor.Collections);

			if (!dryRun && !string.IsNullOrEmpty(dbPath))
			{
				var (version, existing) = CollectionDatabase.Read(dbPath);

				var backupFile = CollectionDatabase.CreateBackup(dbPath);
				if (!string.IsNullOrEmpty(backupFile))
				{
					AnsiConsole.MarkupLine($"[grey]Backup:[/] {backupFile}");
				}

				var finalCollections = new List<CollectionEntry>();

				if (purgeDans)
				{
					var preserved = existing.Where(c => !CollectionDatabase.IsDanCollection(c.Name)).ToList();
					finalCollections.AddRange(preserved);
				}
				else
				{
					finalCollections.AddRange(existing);
				}

				foreach (var kvp in processor.Collections.OrderBy(c => c.Key))
				{
					var distinctHashes = kvp.Value.Distinct().ToList();
					if (distinctHashes.Count == 0) continue;

					var existingCol = finalCollections.FirstOrDefault(c => c.Name.Equals(kvp.Key, StringComparison.OrdinalIgnoreCase));
					if (existingCol != null)
					{
						existingCol.BeatmapHashes = existingCol.BeatmapHashes.Concat(distinctHashes).Distinct().ToList();
					}
					else
					{
						finalCollections.Add(new CollectionEntry
						{
							Name = kvp.Key,
							BeatmapHashes = distinctHashes
						});
					}
				}

				CollectionDatabase.Write(dbPath, version, finalCollections);
				AnsiConsole.MarkupLine($"[bold green]Saved to {dbPath}[/]\n");
			}
		}

		return 0;
	}

	private static void DisplaySummaryTable(ConcurrentDictionary<string, ConcurrentBag<string>> collections)
	{
		var danTree = new Tree("[bold]Collections[/]");

		var globalSkillsets = collections
			.Where(c => c.Key.StartsWith("0") && c.Key.Contains(". "))
			.OrderBy(c => c.Key)
			.ToList();

		if (globalSkillsets.Count > 0)
		{
			var globalNode = danTree.AddNode("[bold]Skillsets[/]");
			foreach (var g in globalSkillsets)
			{
				globalNode.AddNode($"[cyan]{g.Key}[/]: [green]{g.Value.Distinct().Count():N0}[/] maps");
			}
		}

		var danOrder = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
		for (int i = 0; i < DanClassifier.DisplayNames.Length; i++)
		{
			danOrder[DanClassifier.DisplayNames[i]] = i;
			danOrder[DanClassifier.SimpleNames[i]] = i;
		}

		var danParents = collections
			.Where(c => danOrder.ContainsKey(c.Key))
			.OrderBy(c => danOrder[c.Key])
			.ToList();

		var dansNode = danTree.AddNode("[bold]Dans[/]");

		foreach (var parent in danParents)
		{
			var parentName = parent.Key;
			var parentCount = parent.Value.Distinct().Count();

			var danNode = dansNode.AddNode($"[bold]{parentName}[/] [green]({parentCount:N0} maps)[/]");

			var prefix = $"{parentName} - ";
			var subcategories = collections
				.Where(c => c.Key.StartsWith(prefix))
				.OrderBy(c => c.Key)
				.ToList();

			foreach (var sub in subcategories)
			{
				var subName = sub.Key[prefix.Length..];
				var subCount = sub.Value.Distinct().Count();
				danNode.AddNode($"↳ [grey]{subName}:[/] [cyan]{subCount:N0}[/] maps");
			}
		}

		AnsiConsole.Write(danTree);
		AnsiConsole.WriteLine();

		var table = new Table()
			.Border(TableBorder.Rounded)
			.AddColumn("[bold]Dan[/]")
			.AddColumn("[bold]Maps[/]")
			.AddColumn("[bold]Subcategories[/]");

		foreach (var parent in danParents)
		{
			var parentName = parent.Key;
			var parentCount = parent.Value.Distinct().Count();

			var prefix = $"{parentName} - ";
			var subcategories = collections
				.Where(c => c.Key.StartsWith(prefix))
				.OrderByDescending(c => c.Value.Distinct().Count())
				.Select(c => $"{c.Key[prefix.Length..]}: {c.Value.Distinct().Count()}")
				.ToList();

			var subDesc = subcategories.Count > 0
				? string.Join(", ", subcategories)
				: "[grey]None[/]";

			table.AddRow($"[bold]{parentName}[/]", $"[green]{parentCount}[/]", subDesc);
		}

		AnsiConsole.Write(table);
		AnsiConsole.WriteLine();
	}

	private static string? DetectSongsDirectory()
	{
		var envCustom = Environment.GetEnvironmentVariable("OSU_SONGS_PATH");
		if (!string.IsNullOrEmpty(envCustom) && Directory.Exists(envCustom))
			return envCustom;

		var envOsu = Environment.GetEnvironmentVariable("OSU_PATH");
		if (!string.IsNullOrEmpty(envOsu))
		{
			var subSongs = Path.Combine(envOsu, "Songs");
			if (Directory.Exists(subSongs)) return subSongs;
		}

		try
		{
			var osuProcess = Process.GetProcessesByName("osu!").FirstOrDefault() ?? Process.GetProcessesByName("osu").FirstOrDefault();
			if (osuProcess?.MainModule?.FileName is string exePath)
			{
				var dir = Path.GetDirectoryName(exePath);
				if (!string.IsNullOrEmpty(dir))
				{
					var songs = Path.Combine(dir, "Songs");
					if (Directory.Exists(songs)) return songs;
				}
			}
		}
		catch { }

		var candidates = new List<string>();
		var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

		if (OperatingSystem.IsWindows())
		{
			var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
			var appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
			var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
			var programFilesX86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);

			if (!string.IsNullOrEmpty(localAppData)) candidates.Add(Path.Combine(localAppData, "osu!", "Songs"));
			if (!string.IsNullOrEmpty(appData)) candidates.Add(Path.Combine(appData, "osu!", "Songs"));

			foreach (var drive in DriveInfo.GetDrives().Where(d => d.IsReady))
			{
				candidates.Add(Path.Combine(drive.RootDirectory.FullName, "Games", "osu!", "Songs"));
				candidates.Add(Path.Combine(drive.RootDirectory.FullName, "osu!", "Songs"));
			}

			if (!string.IsNullOrEmpty(programFiles)) candidates.Add(Path.Combine(programFiles, "osu!", "Songs"));
			if (!string.IsNullOrEmpty(programFilesX86)) candidates.Add(Path.Combine(programFilesX86, "osu!", "Songs"));
		}
		else if (OperatingSystem.IsMacOS())
		{
			candidates.Add(Path.Combine(home, "Library", "Application Support", "osu", "Songs"));
			candidates.Add("/Applications/osu!.app/Contents/Resources/drive_c/osu/Songs");
			candidates.Add(Path.Combine(home, ".local", "share", "osu-wine", "osu!", "Songs"));
		}
		else
		{
			var xdgData = Environment.GetEnvironmentVariable("XDG_DATA_HOME");
			if (!string.IsNullOrEmpty(xdgData))
				candidates.Add(Path.Combine(xdgData, "osu-wine", "osu!", "Songs"));

			candidates.Add(Path.Combine(home, ".local", "share", "osu-wine", "osu!", "Songs"));
			candidates.Add(Path.Combine(home, "Games", "osu!", "Songs"));
			candidates.Add(Path.Combine(home, ".osu", "Songs"));
			candidates.Add(Path.Combine(home, ".var", "app", "sh.ppy.osu", "data", "osu", "Songs"));
			candidates.Add(Path.Combine(home, ".wine", "drive_c", "osu!", "Songs"));
			candidates.Add(Path.Combine(home, ".wine", "drive_c", "Program Files", "osu!", "Songs"));

			var prefixDir = Path.Combine(home, ".local", "share", "wineprefixes");
			if (Directory.Exists(prefixDir))
			{
				try
				{
					foreach (var p in Directory.EnumerateDirectories(prefixDir))
					{
						candidates.Add(Path.Combine(p, "drive_c", "osu!", "Songs"));
						candidates.Add(Path.Combine(p, "drive_c", "users", Environment.UserName, "AppData", "Local", "osu!", "Songs"));
					}
				}
				catch { }
			}
		}

		return candidates.FirstOrDefault(Directory.Exists);
	}

	private static string? DetectCollectionDb(string? resolvedSongsDir = null)
	{
		if (!string.IsNullOrEmpty(resolvedSongsDir))
		{
			var parent = Directory.GetParent(resolvedSongsDir)?.FullName;
			if (!string.IsNullOrEmpty(parent))
			{
				var siblingDb = Path.Combine(parent, "collection.db");
				if (File.Exists(siblingDb)) return siblingDb;
			}
		}

		var envCustom = Environment.GetEnvironmentVariable("OSU_DB_PATH");
		if (!string.IsNullOrEmpty(envCustom) && File.Exists(envCustom))
			return envCustom;

		var envOsu = Environment.GetEnvironmentVariable("OSU_PATH");
		if (!string.IsNullOrEmpty(envOsu))
		{
			var subDb = Path.Combine(envOsu, "collection.db");
			if (File.Exists(subDb)) return subDb;
		}

		var candidates = new List<string>();
		var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

		if (OperatingSystem.IsWindows())
		{
			var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
			var appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
			var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
			var programFilesX86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);

			if (!string.IsNullOrEmpty(localAppData)) candidates.Add(Path.Combine(localAppData, "osu!", "collection.db"));
			if (!string.IsNullOrEmpty(appData)) candidates.Add(Path.Combine(appData, "osu!", "collection.db"));

			foreach (var drive in DriveInfo.GetDrives().Where(d => d.IsReady))
			{
				candidates.Add(Path.Combine(drive.RootDirectory.FullName, "Games", "osu!", "collection.db"));
				candidates.Add(Path.Combine(drive.RootDirectory.FullName, "osu!", "collection.db"));
			}

			if (!string.IsNullOrEmpty(programFiles)) candidates.Add(Path.Combine(programFiles, "osu!", "collection.db"));
			if (!string.IsNullOrEmpty(programFilesX86)) candidates.Add(Path.Combine(programFilesX86, "osu!", "collection.db"));
		}
		else if (OperatingSystem.IsMacOS())
		{
			candidates.Add(Path.Combine(home, "Library", "Application Support", "osu", "collection.db"));
			candidates.Add("/Applications/osu!.app/Contents/Resources/drive_c/osu/collection.db");
			candidates.Add(Path.Combine(home, ".local", "share", "osu-wine", "osu!", "collection.db"));
		}
		else
		{
			var xdgData = Environment.GetEnvironmentVariable("XDG_DATA_HOME");
			if (!string.IsNullOrEmpty(xdgData))
				candidates.Add(Path.Combine(xdgData, "osu-wine", "osu!", "collection.db"));

			candidates.Add(Path.Combine(home, ".local", "share", "osu-wine", "osu!", "collection.db"));
			candidates.Add(Path.Combine(home, "Games", "osu!", "collection.db"));
			candidates.Add(Path.Combine(home, ".osu", "collection.db"));
			candidates.Add(Path.Combine(home, ".var", "app", "sh.ppy.osu", "data", "osu", "collection.db"));
			candidates.Add(Path.Combine(home, ".wine", "drive_c", "osu!", "collection.db"));
			candidates.Add(Path.Combine(home, ".wine", "drive_c", "Program Files", "osu!", "collection.db"));
		}

		return candidates.FirstOrDefault(File.Exists);
	}
}
