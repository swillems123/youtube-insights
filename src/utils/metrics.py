import logging
logger = logging.getLogger(__name__)


def calculate_engagement_rate(likes, comments, views):
    logger.debug(f"Calculating engagement rate with likes={likes}, comments={comments}, views={views}")
    if views == 0:
        logger.warning("Views is 0, returning engagement rate 0")
        return 0
    result = (likes + comments) / views * 100
    logger.debug(f"Engagement rate calculated: {result}")
    return result


def calculate_average_views(video_views):
    logger.debug(f"Calculating average views for list of size {len(video_views)}")
    if not video_views:
        logger.warning("No video views provided, returning 0")
        return 0
    result = sum(video_views) / len(video_views)
    logger.debug(f"Average views calculated: {result}")
    return result


def calculate_like_to_view_ratio(likes, views):
    logger.debug(f"Calculating like-to-view ratio with likes={likes}, views={views}")
    if views == 0:
        logger.warning("Views is 0, returning ratio 0")
        return 0
    result = likes / views
    logger.debug(f"Like-to-view ratio calculated: {result}")
    return result