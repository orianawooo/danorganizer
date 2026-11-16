using System.Text;
using System.Text.RegularExpressions;

namespace OsuDan;

public class CollectionEntry
{
	public string Name { get; set; } = string.Empty;
	public List<string> BeatmapHashes { get; set; } = new();
}

public static class CollectionDatabase
{
	private static readonly Regex DanPattern = new(
		@"^(Dan\s+\d+|Dan\s+\d+\s+-\s+.*|Dan\s+.*|0[1-7]\.\s+.*)$",
		RegexOptions.Compiled | RegexOptions.IgnoreCase);

	public static (int Version, List<CollectionEntry> Collections) Read(string path)
	{
		var collections = new List<CollectionEntry>();
		if (!File.Exists(path))
			return (20260924, collections);

		using var stream = File.OpenRead(path);
		using var reader = new BinaryReader(stream, Encoding.UTF8);

		var version = reader.ReadInt32();
		var count = reader.ReadInt32();

		for (var i = 0; i < count; i++)
		{
			var name = ReadOsuString(reader);
			var mapCount = reader.ReadInt32();
			var hashes = new List<string>(mapCount);

			for (var j = 0; j < mapCount; j++)
			{
				var hash = ReadOsuString(reader);
				if (!string.IsNullOrEmpty(hash))
					hashes.Add(hash.ToLowerInvariant());
			}

			collections.Add(new CollectionEntry
			{
				Name = name,
				BeatmapHashes = hashes
			});
		}

		return (version, collections);
	}

	public static string CreateBackup(string path)
	{
		if (!File.Exists(path))
			return string.Empty;

		var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
		var backupPath = $"{path}.bak_{timestamp}";
		File.Copy(path, backupPath, true);
		return backupPath;
	}

	public static void Write(string path, int version, List<CollectionEntry> collections)
	{
		var dir = Path.GetDirectoryName(path) ?? ".";
		var tempPath = Path.Combine(dir, $"collection_temp_{Guid.NewGuid():N}.db");

		using (var stream = File.Create(tempPath))
		using (var writer = new BinaryWriter(stream, Encoding.UTF8))
		{
			writer.Write(version);
			writer.Write(collections.Count);

			foreach (var col in collections)
			{
				WriteOsuString(writer, col.Name);
				writer.Write(col.BeatmapHashes.Count);

				foreach (var hash in col.BeatmapHashes)
				{
					WriteOsuString(writer, hash.ToLowerInvariant());
				}
			}
		}

		File.Move(tempPath, path, true);
	}

	public static bool IsDanCollection(string name)
	{
		if (string.IsNullOrWhiteSpace(name))
			return false;

		return DanPattern.IsMatch(name.Trim());
	}

	private static string ReadOsuString(BinaryReader reader)
	{
		var flag = reader.ReadByte();
		if (flag == 0x00) return string.Empty;
		if (flag != 0x0b) throw new InvalidDataException($"Invalid string flag: 0x{flag:X2}");

		var length = ReadULEB128(reader);
		var bytes = reader.ReadBytes((int)length);
		return Encoding.UTF8.GetString(bytes);
	}

	private static void WriteOsuString(BinaryWriter writer, string value)
	{
		if (string.IsNullOrEmpty(value))
		{
			writer.Write((byte)0x00);
			return;
		}

		writer.Write((byte)0x0b);
		var bytes = Encoding.UTF8.GetBytes(value);
		WriteULEB128(writer, (uint)bytes.Length);
		writer.Write(bytes);
	}

	private static uint ReadULEB128(BinaryReader reader)
	{
		uint result = 0;
		var shift = 0;

		while (true)
		{
			var b = reader.ReadByte();
			result |= (uint)(b & 0x7F) << shift;
			if ((b & 0x80) == 0) break;
			shift += 7;
		}

		return result;
	}

	private static void WriteULEB128(BinaryWriter writer, uint value)
	{
		do
		{
			var b = (byte)(value & 0x7F);
			value >>= 7;
			if (value != 0) b |= 0x80;
			writer.Write(b);
		} while (value != 0);
	}
}
