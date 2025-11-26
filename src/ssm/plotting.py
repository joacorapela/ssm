
def add_events_vlines(fig, events_df):
    n_events = events_df.shape[0]
    for i in range(n_events):
        fig.add_vline(x=events_df.iloc[i]["event_time"],
                      line_dash=events_df.iloc[i]["event_line_type"],
                      line_color=events_df.iloc[i]["event_color"])
