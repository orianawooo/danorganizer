using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OsuDan;

public class DanMapCacheEntry
{
	[JsonPropertyName("md5")] public string Md5 { get; set; } = string.Empty;
	[JsonPropertyName("last_modified")] public long LastModifiedUtc { get; set; }
	[JsonPropertyName("dan_index")] public int DanIndex { get; set; }
	[JsonPropertyName("dan_name")] public string DanName { get; set; } = string.Empty;
	[JsonPropertyName("skillset")] public string DominantSkillset { get; set; } = string.Empty;
}

public class BatchProgress
{
	public int TotalFiles { get; set; }
	public int ProcessedFiles { get; set; }
	public int ValidFound { get; set; }
	public int SkippedNon4K { get; set; }
	public int CachedHits { get; set; }
	public string CurrentMapName { get; set; } = string.Empty;
}

public class BatchProcessor
{
	private readonly MsdCalculator _msdCalculator;
	private readonly DanClassifier _classifier;
	private readonly string _cacheFilePath;
	private readonly bool _zeroPadded;
	private readonly ConcurrentDictionary<string, DanMapCacheEntry> _cache = new();

	public ConcurrentDictionary<string, ConcurrentBag<string>> Collections { get; } = new();

	public BatchProcessor(
		MsdCalculator msdCalculator,
		DanClassifier classifier,
		string? cacheFilePath = null,
		bool zeroPadded = true)
	{
		_msdCalculator = msdCalculator;
		_classifier = classifier;
		_cacheFilePath = ResolveCachePath(cacheFilePath);
		_zeroPadded = zeroPadded;
		LoadCache();
	}

	private static string ResolveCachePath(string? customPath)
	{
		if (!string.IsNullOrEmpty(customPath))
			return customPath;

		var baseDir = AppDomain.CurrentDomain.BaseDirectory;
		var projectCache = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "dan_cache.json"));
		if (File.Exists(projectCache))
			return projectCache;

		var binCache = Path.Combine(baseDir, "dan_cache.json");
		if (File.Exists(binCache))
			return binCache;

		var cwdCache = Path.Combine(Directory.GetCurrentDirectory(), "dan_cache.json");
		if (File.Exists(cwdCache))
			return cwdCache;

		return binCache;
	}

	private void LoadCache()
	{
		if (!File.Exists(_cacheFilePath))
			return;

		try
		{
			var json = File.ReadAllText(_cacheFilePath);
			var loaded = JsonSerializer.Deserialize<Dictionary<string, DanMapCacheEntry>>(json);
			if (loaded != null)
			{
				foreach (var kvp in loaded)
					_cache[kvp.Key] = kvp.Value;
			}
		}
		catch { }
	}

	public void SaveCache()
	{
		try
		{
			var json = JsonSerializer.Serialize(_cache, new JsonSerializerOptions { WriteIndented = false });
			File.WriteAllText(_cacheFilePath, json);
		}
		catch { }
	}

	public async Task ProcessSongsAsync(
		string songsDirectory,
		int maxConcurrency,
		IProgress<BatchProgress>? progress = null,
		int? testLimit = null,
		CancellationToken cancellationToken = default)
	{
		if (!Directory.Exists(songsDirectory))
			throw new DirectoryNotFoundException($"Songs directory not found: {songsDirectory}");

		var files = Directory.EnumerateFiles(songsDirectory, "*.osu", SearchOption.AllDirectories);
		if (testLimit.HasValue && testLimit.Value > 0)
			files = files.Take(testLimit.Value);

		var fileList = files.ToList();
		var total = fileList.Count;

		var processedCount = 0;
		var validCount = 0;
		var skippedCount = 0;
		var cachedCount = 0;

		var parallelOptions = new ParallelOptions
		{
			MaxDegreeOfParallelism = maxConcurrency > 0 ? maxConcurrency : Environment.ProcessorCount,
			CancellationToken = cancellationToken
		};

		await Parallel.ForEachAsync(fileList, parallelOptions, async (filePath, ct) =>
		{
			if (ct.IsCancellationRequested) return;

			var mapFileName = Path.GetFileName(filePath);

			if (!BeatmapData.QuickCheckIs4K(filePath))
			{
				Interlocked.Increment(ref skippedCount);
				var p = Interlocked.Increment(ref processedCount);
				if (p % 100 == 0 || p == total)
				{
					progress?.Report(new BatchProgress
					{
						TotalFiles = total,
						ProcessedFiles = p,
						ValidFound = validCount,
						SkippedNon4K = skippedCount,
						CachedHits = cachedCount,
						CurrentMapName = mapFileName
					});
				}
				return;
			}

			string md5;
			long lastModified;
			byte[] fileBytes;
			try
			{
				var fileInfo = new FileInfo(filePath);
				lastModified = fileInfo.LastWriteTimeUtc.Ticks;
				fileBytes = await File.ReadAllBytesAsync(filePath, ct);
				md5 = Convert.ToHexString(MD5.HashData(fileBytes)).ToLowerInvariant();
			}
			catch
			{
				Interlocked.Increment(ref skippedCount);
				Interlocked.Increment(ref processedCount);
				return;
			}

			if (_cache.TryGetValue(filePath, out var cached) && cached.LastModifiedUtc == lastModified && cached.Md5 == md5)
			{
				Interlocked.Increment(ref cachedCount);
				Interlocked.Increment(ref validCount);
				var danName = _zeroPadded
					? DanClassifier.DisplayNames[Math.Clamp(cached.DanIndex, 0, DanClassifier.DisplayNames.Length - 1)]
					: DanClassifier.SimpleNames[Math.Clamp(cached.DanIndex, 0, DanClassifier.SimpleNames.Length - 1)];
				AddMapToCollections(danName, cached.DominantSkillset, md5);

				var p = Interlocked.Increment(ref processedCount);
				if (p % 50 == 0 || p == total)
				{
					progress?.Report(new BatchProgress
					{
						TotalFiles = total,
						ProcessedFiles = p,
						ValidFound = validCount,
						SkippedNon4K = skippedCount,
						CachedHits = cachedCount,
						CurrentMapName = mapFileName
					});
				}
				return;
			}

			BeatmapData beatmap;
			double interludeRating;
			double sunnyRating;

			try
			{
				beatmap = BeatmapData.Parse(filePath, fileBytes);
				interludeRating = InterludeCalculator.Calculate(beatmap);
				sunnyRating = SunnyCalculator.Calculate(beatmap);
			}
			catch
			{
				Interlocked.Increment(ref skippedCount);
				Interlocked.Increment(ref processedCount);
				return;
			}

			var msdResult = await _msdCalculator.CalculateAsync(filePath);
			if (msdResult == null)
			{
				Interlocked.Increment(ref skippedCount);
				Interlocked.Increment(ref processedCount);
				return;
			}

			var classification = _classifier.Classify(msdResult, interludeRating, sunnyRating, _zeroPadded);

			if (classification != null)
			{
				Interlocked.Increment(ref validCount);
				AddMapToCollections(classification.DanName, classification.DominantSkillset, md5);

				_cache[filePath] = new DanMapCacheEntry
				{
					Md5 = md5,
					LastModifiedUtc = lastModified,
					DanIndex = classification.DanIndex,
					DanName = classification.DanName,
					DominantSkillset = classification.DominantSkillset
				};
			}
			else
			{
				Interlocked.Increment(ref skippedCount);
			}

			var count = Interlocked.Increment(ref processedCount);
			if (count % 25 == 0 || count == total)
			{
				progress?.Report(new BatchProgress
				{
					TotalFiles = total,
					ProcessedFiles = count,
					ValidFound = validCount,
					SkippedNon4K = skippedCount,
					CachedHits = cachedCount,
					CurrentMapName = beatmap.DisplayName
				});
			}
		});

		SaveCache();
	}

	private void AddMapToCollections(string danName, string dominantSkillset, string md5)
	{
		Collections.GetOrAdd(danName, _ => new ConcurrentBag<string>()).Add(md5);

		var danSkillset = $"{danName} - {dominantSkillset}";
		Collections.GetOrAdd(danSkillset, _ => new ConcurrentBag<string>()).Add(md5);

		var globalSkillset = GetGlobalSkillsetName(dominantSkillset);
		if (!string.IsNullOrEmpty(globalSkillset))
		{
			Collections.GetOrAdd(globalSkillset, _ => new ConcurrentBag<string>()).Add(md5);
		}
	}

	private static string GetGlobalSkillsetName(string skillset)
	{
		return skillset.ToLowerInvariant() switch
		{
			"chordjack" => "01. Chordjack",
			"handstream" => "02. Handstream",
			"jackspeed" => "03. Jackspeed",
			"jumpstream" => "04. Jumpstream",
			"stamina" => "05. Stamina",
			"stream" => "06. Stream",
			"technical" => "07. Technical",
			_ => $"08. {skillset}"
		};
	}
}
