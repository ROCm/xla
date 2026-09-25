# frozen_string_literal: true

# Incremental rebuilds always emit `_config.yml` `baseurl` in `relative_url`.
# Mount the same `_site` files at "/" and "/xla/scaling-book" so local preview
# works with either URL. `jekyll build` never calls this method.

require "webrick"
require "jekyll/commands/serve"
require "jekyll/commands/serve/servlet"

module ScaleYourAmd
  module ServeBothPrefixes
    def start_up_webrick(opts, destination)
      @reload_reactor.start(opts) if opts["livereload"]

      @server = WEBrick::HTTPServer.new(webrick_opts(opts)).tap { |o| o.unmount("") }
      servlet = Jekyll::Commands::Serve::Servlet
      ["/xla/scaling-book", "/"].each do |prefix|
        @server.mount(prefix, servlet, destination, file_handler_opts)
      end

      Jekyll.logger.info "Server address:", server_address(@server, opts)
      Jekyll.logger.info "Also serving:", format_url(
        @server.config[:SSLEnable],
        @server.config[:BindAddress],
        @server.config[:Port],
        nil
      )
      launch_browser @server, opts if opts["open_url"]
      boot_or_detach @server, opts
    end
  end
end

Jekyll::Commands::Serve.singleton_class.prepend(ScaleYourAmd::ServeBothPrefixes)
