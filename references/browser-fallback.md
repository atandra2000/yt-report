# Browser fallback (load only when `download_video` itself fails)

Captions being rate-limited does **not** imply the video is un-downloadable — in testing, yt-dlp kept working through a caption block. Reach for the browser (`ego-browser`, a real logged-in Chromium) only when the download fails, or when you need to know *why*.

```js
const task = await taskSpace("yt probe");
const page = task.page("p1");
await page.goto(VIDEO_URL, { timeout: 45000 });
const r = await page.evaluate(() => {
  const pr = window.ytInitialPlayerResponse || {};
  const tracks = pr?.captions?.playerCaptionsTracklistRenderer?.captionTracks || [];
  return {
    playability: pr?.playabilityStatus?.status,   // "OK" = watchable, "LOGIN_REQUIRED" = age/sign-in wall
    reason: pr?.playabilityStatus?.reason,
    captionLangs: tracks.map((t) => `${t.languageCode}/${t.kind || "manual"}`),
  };
});
console.log(r);
await task.finish({ keep: [] });
```

**What it is good for:**
- A **playability oracle** — `LOGIN_REQUIRED` / `UNPLAYABLE` says the video is gated, which no API error message states as clearly.
- **Listing which caption tracks exist**, independent of whether the transcript API is currently blocked.
- **Reading the video when frames are not enough** — the player is live, so a genuine frame-scrubbing check is possible.
- **A stream URL** — `ytInitialPlayerResponse.streamingData` exposes direct signed `googlevideo.com` URLs (measured: one 360p progressive MP4 plus 112 adaptive formats). Hand that URL to `curl`, a normal HTTP download, rather than fetching bytes through the page.

**What it does not do** (verified, not assumed):
- **It does not bypass a caption block.** Pulling the track list through the browser succeeds; fetching `baseUrl + "&fmt=json3"` still returned **429**. That block is per-IP, not per-tool. Do not spend turns retrying this route.
- **It is not a fast media transport.** `page.fetch(url, { saveAs })` on a 12 MB stream ran past five minutes. Use the browser for the *URL*, `curl` for the bytes.
- **It can wedge.** A large in-page fetch left the task space unable to complete a later navigation, and a fresh space did not recover it. If the space stops responding, stop and ask the user rather than burning turns on retries.

Report the distinction honestly: a video the browser can play but that yields no transcript means the ASR path is the answer, not a browser retry.
