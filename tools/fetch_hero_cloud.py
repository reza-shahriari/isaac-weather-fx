"""Download the Walt Disney Animation Studios cloud and set it up as a weather-fx hero cloud.

    python tools/fetch_hero_cloud.py                       # download, keep the eighth resolution
    python tools/fetch_hero_cloud.py --resolution quarter  # sharper (about 100M voxels, needs RAM)
    python tools/fetch_hero_cloud.py --zip ~/Downloads/wdas_cloud.zip   # already downloaded

Plain Python, no Isaac Sim needed. The package is a single large zip (every resolution plus
renders), so the download takes a while; only the resolution you pick is kept, in
``~/.cache/weather_fx/clouds``. The script prints the line that turns the hero clouds on.

**License.** The cloud is Copyright 2017 Disney Enterprises, Inc., licensed under Creative
Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0). You may use it for anything,
commercial work included, as long as you credit it; ShareAlike means adaptations of the data you
distribute carry the same license. An ATTRIBUTION.txt is written next to the file. Source:
https://www.disneyanimation.com/resources/clouds/
"""
import argparse
import pathlib
import shutil
import sys
import urllib.request
import zipfile

URL = "https://assets.disneyanimation.com/wdas_cloud.zip"
RESOLUTIONS = ("sixteenth", "eighth", "quarter", "half", "full")
ATTRIBUTION = """\
Walt Disney Animation Studios Cloud Data Set
Copyright 2017 Disney Enterprises, Inc.
Licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License:
https://creativecommons.org/licenses/by-sa/3.0/
Source: https://www.disneyanimation.com/resources/clouds/
The cloud model is based on a photograph by Kevin Udy (Colorado Clouds Blog), same license.

Credit it wherever the cloud, or an image rendered with it, is shown.
"""


def member_name(resolution: str) -> str:
    return "wdas_cloud.vdb" if resolution == "full" else f"wdas_cloud_{resolution}.vdb"


def download(url: str, target: pathlib.Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({100 * done / total:5.1f} %)",
                      end="", flush=True)
            else:
                print(f"\r  {done / 1e6:8.1f} MB", end="", flush=True)
    print()
    partial.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resolution", choices=RESOLUTIONS, default="eighth")
    parser.add_argument("--zip", help="use this already-downloaded wdas_cloud.zip")
    parser.add_argument("--dest", default=str(pathlib.Path.home() / ".cache" / "weather_fx" / "clouds"))
    parser.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip")
    args = parser.parse_args()

    dest = pathlib.Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    wanted = member_name(args.resolution)
    target = dest / wanted
    if not target.is_file():
        archive = pathlib.Path(args.zip).expanduser() if args.zip else dest / "wdas_cloud.zip"
        if not archive.is_file():
            print(f"downloading {URL}\n  (the whole package: every resolution plus renders; "
                  "this is large and takes a while)")
            download(URL, archive)
        with zipfile.ZipFile(archive) as package:
            members = [m for m in package.namelist() if m.endswith("/" + wanted) or m == wanted]
            if not members:
                print(f"{wanted} is not in {archive}; it holds:", *package.namelist(), sep="\n  ")
                return 1
            print(f"extracting {members[0]}")
            with package.open(members[0]) as source, open(target, "wb") as out:
                shutil.copyfileobj(source, out, 1 << 20)
        if not args.zip and not args.keep_zip:
            archive.unlink()
    (dest / "ATTRIBUTION.txt").write_text(ATTRIBUTION)
    print(f"\nhero cloud: {target}\nattribution: {dest / 'ATTRIBUTION.txt'}\n")
    print("Turn it on (Script Editor, path tracer):\n")
    print("    from weather_fx import api")
    print(f"    api.get_controller().set_clouds(enabled=True, hero_vdb={str(target)!r}, hero_count=3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
