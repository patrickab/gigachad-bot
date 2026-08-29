use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, RunEvent, State};
use tauri_plugin_shell::{process::{CommandChild, CommandEvent}, ShellExt};

struct BackendState {
    base_url: String,
    child: Mutex<Option<CommandChild>>,
}

#[derive(Deserialize)]
struct SidecarEvent {
    event: String,
    port: u16,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendInfo {
    base_url: String,
}

#[tauri::command]
fn backend_info(state: State<'_, BackendState>) -> BackendInfo {
    BackendInfo { base_url: state.base_url.clone() }
}

async fn start_sidecar(app: &AppHandle) -> Result<(String, CommandChild), String> {
    // The sidecar ships as a PyInstaller onedir bundle (executable + _internal/
    // deps) rather than a single file, so it's a resource, not `externalBin`.
    // PyInstaller already marks this executable, and bundling preserves that bit;
    // an AppImage's squashfs mount is read-only, so we can't chmod it here anyway.
    let sidecar_path = app
        .path()
        .resolve("sidecar/gigachad-sidecar", tauri::path::BaseDirectory::Resource)
        .map_err(|error| format!("could not resolve Python sidecar resource: {error}"))?;

    eprintln!("[gigachad-bot] resolved sidecar path: {}", sidecar_path.display());

    // The sidecar writes chat histories, uploads, etc. under GIGACHAD_BASE_DIR
    // (defaults to CWD, which is read-only inside a mounted AppImage). A value
    // already present in the launch environment wins, so the desktop app can
    // share a data dir with the webapp.
    let base_dir = match std::env::var_os("GIGACHAD_BASE_DIR") {
        Some(dir) => std::path::PathBuf::from(dir),
        None => app
            .path()
            .app_data_dir()
            .map_err(|error| format!("could not resolve app data dir: {error}"))?,
    };
    std::fs::create_dir_all(&base_dir)
        .map_err(|error| format!("could not create app data dir: {error}"))?;

    let command = app
        .shell()
        .command(&sidecar_path)
        .env("GIGACHAD_BASE_DIR", &base_dir)
        .env(
            "GIGACHAD_CORS_ORIGINS",
            "http://127.0.0.1:2999,http://localhost:2999,http://tauri.localhost,https://tauri.localhost,tauri://localhost",
        )
        .args(["--host", "127.0.0.1", "--port", "0"]);
    let (mut events, child) = command
        .spawn()
        .map_err(|error| format!("could not start Python sidecar: {error}"))?;

    let mut stderr_tail: Vec<String> = Vec::new();
    while let Some(event) = events.recv().await {
        match event {
            CommandEvent::Stdout(bytes) => {
                let line = String::from_utf8_lossy(&bytes);
                eprintln!("[gigachad-sidecar stdout] {line}");
                if let Ok(ready) = serde_json::from_str::<SidecarEvent>(&line) {
                    if ready.event == "ready" {
                        tauri::async_runtime::spawn(async move {
                            while events.recv().await.is_some() {}
                        });
                        return Ok((format!("http://127.0.0.1:{}", ready.port), child));
                    }
                }
            }
            CommandEvent::Stderr(bytes) => {
                let line = String::from_utf8_lossy(&bytes).into_owned();
                eprintln!("[gigachad-sidecar stderr] {line}");
                stderr_tail.push(line);
            }
            CommandEvent::Error(error) => return Err(format!("Python sidecar failed: {error}")),
            CommandEvent::Terminated(payload) => {
                return Err(format!(
                    "Python sidecar stopped before readiness: {payload:?}\nstderr:\n{}",
                    stderr_tail.join("\n")
                ))
            }
            _ => {}
        }
    }

    Err("Python sidecar closed its output before readiness".to_owned())
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let (base_url, child) = tauri::async_runtime::block_on(start_sidecar(&app.handle()))
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
            app.manage(BackendState { base_url, child: Mutex::new(Some(child)) });
            app.get_webview_window("main").expect("main window must exist").show()?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![backend_info])
        .build(tauri::generate_context!())
        .expect("error while building GigaChat Bot");

    app.run(|app_handle, event| {
        if matches!(event, RunEvent::ExitRequested { .. }) {
            if let Some(child) = app_handle.state::<BackendState>().child.lock().expect("sidecar lock poisoned").take() {
                let _ = child.kill();
            }
        }
    });
}
