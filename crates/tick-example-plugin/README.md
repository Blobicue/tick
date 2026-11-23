# tick-example-plugin

The reference plugin: `GET /plugin/status` (engine snapshot as JSON)
and `POST /plugin/ask` (authed one-shot generation). Copy this crate
as a template for your own — mount it with
`tick_server::server::serve_with(bind, app, vec![Box::new(StatusBoard)])`.

The wedge-free proof test in tick-server mounts this plugin and shows
its routes answering in microseconds during an active generation.
